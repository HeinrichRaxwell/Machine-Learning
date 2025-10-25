# ===== Pertemuan 5 — Modeling (robust untuk dataset kecil) =====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# ---------- Langkah 1: Muat data & split (70/15/15) ----------
df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# stratify sekali (train vs temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# cari seed supaya y_test punya dua kelas → ROC bisa digambar
seed_found = None
for rs in range(500):
    X_val_try, X_test_try, y_val_try, y_test_try = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=rs
    )
    if len(set(y_test_try)) == 2:
        seed_found = rs
        X_val, X_test, y_val, y_test = X_val_try, X_test_try, y_val_try, y_test_try
        break

print(f"seed_found (split kedua): {seed_found}")
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Train labels:\n", y_train.value_counts())
print("Val labels:\n",   y_val.value_counts())
print("Test labels:\n",  y_test.value_counts())

# ---------- Langkah 2: Baseline LogReg (pipeline) ----------
num_cols = X_train.select_dtypes(include="number").columns
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]), num_cols)
], remainder="drop")

pipe_lr = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
pipe_lr.fit(X_train, y_train)
y_val_lr = pipe_lr.predict(X_val)
print("\nBaseline (LogReg) F1(val):", f1_score(y_val, y_val_lr, average="macro"))
print(classification_report(y_val, y_val_lr, digits=3))

# ---------- Langkah 3: Model Alternatif — RandomForest ----------
pipe_rf = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
    ))
])
pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("\nRandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))

# ---------- Langkah 4: Tuning ringkas (GridSearch 2-fold) ----------
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
param = {
  "clf__max_depth": [None, 12, 20, 30],
  "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))

# ---------- Langkah 5: Evaluasi Akhir (TEST) ----------
# pilih final model: pakai yang terbaik (RF tuned); kalau mau, bandingkan dengan LR
final_model = best_rf

y_test_pred = final_model.predict(X_test)
print("\n=== TEST SET ===")
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix (test):\n", cm)

# simpan confusion matrix ke gambar
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
    plt.tight_layout(); plt.savefig("cm_test_p5.png", dpi=120); plt.close()
    print("Saved: cm_test_p5.png")

save_cm(cm, classes=["0","1"], title="Confusion Matrix (test)", filename="cm_test_p5.png")

# ROC-AUC (test) + kurva
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_test_proba)
    print("ROC-AUC(test):", auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--', label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (test)")
    plt.legend(); plt.tight_layout(); plt.savefig("roc_test_p5.png", dpi=120); plt.close()
    print("Saved: roc_test_p5.png")

# ---------- Langkah 6 (Opsional): Simpan Model ----------
# import joblib
# import joblib; joblib.dump(final_model, "model_p5.pkl")
# print("Model tersimpan ke model_p5.pkl")
