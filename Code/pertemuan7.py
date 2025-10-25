# ===== Pertemuan 7 — ANN untuk Klasifikasi (robust untuk dataset kecil) =====
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import layers

# ---------- Seed utk reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Langkah 1 — Load & Split (tanpa leakage) ----------
assert os.path.exists("processed_kelulusan.csv"), "Taruh processed_kelulusan.csv di folder ini."
df = pd.read_csv("processed_kelulusan.csv")

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# 70/30 stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)

# cari seed agar test berisi dua kelas (0 & 1) → ROC/PR valid
seed_found = None
for rs in range(500):
    X_val_try, X_test_try, y_val_try, y_test_try = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=rs
    )
    if len(set(y_test_try)) == 2:
        seed_found = rs
        X_val, X_test, y_val, y_test = X_val_try, X_test_try, y_val_try, y_test_try
        break

print(f"[INFO] seed split kedua (val/test): {seed_found}")
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Label count — train:\n", y_train.value_counts())
print("Label count — val:\n",   y_val.value_counts())
print("Label count — test:\n",  y_test.value_counts())

# ---------- Standardize (fit di train saja) ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ---------- Langkah 2 — Bangun Model ANN ----------
model = keras.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="AUC")]
)

model.summary()

# ---------- Langkah 3 — Training + EarlyStopping ----------
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# dataset kecil → batch_size kecil juga
history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    epochs=200, batch_size=4,
    callbacks=[es], verbose=1
)

# ---------- Langkah 4 — Evaluasi Test (acc/AUC Keras) ----------
loss, acc, auc = model.evaluate(X_test_s, y_test, verbose=0)
print(f"\n=== TEST (Keras) ===\nLoss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

# Probabilitas & pred default threshold 0.5
y_proba_test = model.predict(X_test_s).ravel()
y_pred_test_050 = (y_proba_test >= 0.5).astype(int)

print("\nConfusion Matrix (test, thr=0.5):")
print(confusion_matrix(y_test, y_pred_test_050))
print("\nClassification Report (test, thr=0.5):")
print(classification_report(y_test, y_pred_test_050, digits=3))

# ---------- (Opsional kuat) Tuning Threshold pakai VAL untuk F1 ----------
y_proba_val = model.predict(X_val_s).ravel()
prec, rec, thr = precision_recall_curve(y_val, y_proba_val)
# F1 = 2PR/(P+R), hindari div0
f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
best_idx = int(np.argmax(f1s))
best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
print(f"\n[VAL] Best threshold by F1: {best_thr:.4f} | F1(val)~{f1s[best_idx]:.3f}")

# Evaluasi test pake threshold terbaik dari VAL
y_pred_test_best = (y_proba_test >= best_thr).astype(int)
print("\n=== TEST (Threshold from VAL) ===")
print("F1(test):", f1_score(y_test, y_pred_test_best, average="macro"))
print(classification_report(y_test, y_pred_test_best, digits=3))
print("Confusion Matrix (test, best thr):")
cm_best = confusion_matrix(y_test, y_pred_test_best)
print(cm_best)

# ROC-AUC sklearn (pastikan 2 kelas di test)
if len(set(y_test)) == 2:
    auc_sklearn = roc_auc_score(y_test, y_proba_test)
    print("ROC-AUC(test, sklearn):", round(auc_sklearn, 4))

# ---------- Langkah 5 — Visualisasi Learning Curve ----------
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve (ANN)")
plt.tight_layout(); plt.savefig("learning_curve_p7.png", dpi=120); plt.close()
print("Saved: learning_curve_p7.png")

# ---------- Simpan ROC & PR curve (pakai test set) ----------
if len(set(y_test)) == 2:
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    plt.figure(); plt.plot(fpr, tpr, label="ROC (test)")
    plt.plot([0,1],[0,1],'--', label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (test, ANN)")
    plt.legend(); plt.tight_layout(); plt.savefig("roc_ann_p7.png", dpi=120); plt.close()
    print("Saved: roc_ann_p7.png")

    # PR
    prec_t, rec_t, _ = precision_recall_curve(y_test, y_proba_test)
    plt.figure(); plt.plot(rec_t, prec_t)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (test, ANN)")
    plt.tight_layout(); plt.savefig("pr_ann_p7.png", dpi=120); plt.close()
    print("Saved: pr_ann_p7.png")
else:
    print("ROC/PR test di-skip (test hanya 1 kelas).")

# ---------- Simpan Model + Scaler ----------
model.save("ann_p7.h5")
joblib.dump(scaler, "scaler_p7.pkl")
print("\nSaved model: ann_p7.h5 | scaler: scaler_p7.pkl")

# ---------- (Opsional) Quick Inference Demo ----------
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14,
    "IPK_x_Study": 3.4*7
}])
sample_s = scaler.transform(sample)
proba_sample = float(model.predict(sample_s).ravel()[0])
pred_sample  = int(proba_sample >= best_thr)
print(f"\nContoh prediksi sample → proba={proba_sample:.3f} | thr={best_thr:.3f} | pred={pred_sample}")
