# ===== Pertemuan 6 — Random Forest untuk Klasifikasi (Pilihan A) =====
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import joblib

# ---------- Langkah 1 — Muat Data (processed_kelulusan.csv) ----------
assert os.path.exists("processed_kelulusan.csv"), "Taruh processed_kelulusan.csv di folder ini."
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# split 70/15/15 — stratify sekali (train vs temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# cari seed supaya TEST punya 2 kelas → ROC/PR bisa diplot
seed_found = None
for rs in range(500):
    X_val_try, X_test_try, y_val_try, y_test_try = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=rs
    )
    if len(set(y_test_try)) == 2:
        seed_found = rs
        X_val, X_test, y_val, y_test = X_val_try, X_test_try, y_val_try, y_test_try
        break

print(f"[INFO] seed split kedua: {seed_found}")
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Label count — train:\n", y_train.value_counts())
print("Label count — val:\n",   y_val.value_counts())
print("Label count — test:\n",  y_test.value_counts())

# ---------- Langkah 2 — Pipeline & Baseline Random Forest ----------
num_cols = X_train.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ]), num_cols)
], remainder="drop")

rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)
pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("\nBaseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))

# ---------- Langkah 3 — Validasi Silang (pakai 2-fold biar aman) ----------
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print("CV F1-macro (train):", scores.mean(), "±", scores.std())

# ---------- Langkah 4 — Tuning Ringkas (GridSearch) ----------
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
best_model = gs.best_estimator_
y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ---------- Langkah 5 — Evaluasi Akhir (Test Set) ----------
final_model = best_model  # kalau baseline lebih baik, ganti ke pipe
y_test_pred = final_model.predict(X_test)

print("\n=== TEST EVALUATION ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (test):\n", cm)

# Simpan Confusion Matrix (gambar)
def save_cm(cm, classes, title, filename):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True', title=title)
    thr = cm.max()/2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black')
    plt.tight_layout(); plt.savefig(filename, dpi=120); plt.close()
    print("Saved:", filename)

save_cm(cm, classes=["0","1"], title="Confusion Matrix (test)", filename="cm_test_p6.png")

# ROC-AUC & PR Curve
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_test_proba)
    print("ROC-AUC(test):", auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--', label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (test)")
    plt.legend(); plt.tight_layout(); plt.savefig("roc_test_p6.png", dpi=120); plt.close()
    print("Saved: roc_test_p6.png")

    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (test)")
    plt.tight_layout(); plt.savefig("pr_test_p6.png", dpi=120); plt.close()
    print("Saved: pr_test_p6.png")
else:
    print("Model tidak punya predict_proba → ROC/PR di-skip.")

# ---------- Langkah 6 — Pentingnya Fitur ----------
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out()
    # rapihin nama (hapus prefix "num__")
    feat_names = [n.replace("num__", "") for n in fn]
    top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop feature importance:")
    for name, val in top[:10]:
        print(f"{name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)

# ---------- Langkah 7 — Simpan Model ----------
joblib.dump(final_model, "rf_model.pkl")
print("\nModel disimpan sebagai rf_model.pkl")

# ---------- Langkah 8 — Cek Inference Lokal ----------
sample = pd.DataFrame([{
  "IPK": 3.4,
  "Jumlah_Absensi": 4,
  "Waktu_Belajar_Jam": 7,
  "Rasio_Absensi": 4/14,
  "IPK_x_Study": 3.4*7
}])
pred = int(final_model.predict(sample)[0])
proba = float(final_model.predict_proba(sample)[:,1][0]) if hasattr(final_model, "predict_proba") else None
print("Prediksi sample:", pred, "| proba:", proba)
