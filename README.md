# 🎓 Prediksi Kelulusan Mahasiswa — Machine Learning Project (Unpam)

Proyek ini berisi serangkaian latihan praktikum mata kuliah Machine Learning di Universitas Pamulang (UNPAM), berfokus pada pembuatan model prediksi kelulusan mahasiswa berdasarkan data **IPK, kehadiran, dan waktu belajar.**

Setiap pertemuan merepresentasikan tahapan berbeda dalam pipeline Machine Learning — dari *data preparation*, *modeling*, hingga *neural network*.

---

## 📘 Tahapan Tiap Pertemuan

### **Pertemuan 4 — Data Preparation**
- Membersihkan data dari missing value & duplikasi.
- Eksplorasi data (EDA): boxplot, histogram, scatter plot, heatmap korelasi.
- Feature engineering:  
  - `Rasio_Absensi = Jumlah_Absensi / 14`  
  - `IPK_x_Study = IPK * Waktu_Belajar_Jam`
- Split data → 70% train, 15% val, 15% test  
- Hasil: `processed_kelulusan.csv`

### **Pertemuan 5 — Modeling (Baseline vs Random Forest)**
- Baseline: Logistic Regression.
- Model alternatif: Random Forest.
- Evaluasi: F1, Precision, Recall, ROC-AUC, Confusion Matrix.
- Tuning: `GridSearchCV`
- Output: `model_p5.pkl`

### **Pertemuan 6 — Random Forest (Optimasi Final)**
- Validasi silang (`StratifiedKFold`)
- GridSearch (`max_depth`, `min_samples_split`)
- Visualisasi: ROC, PR, Confusion Matrix.
- Feature Importance → `IPK_x_Study`, `IPK`, `Rasio_Absensi`
- Output: `rf_model.pkl`

### **Pertemuan 7 — Artificial Neural Network (ANN)**
- Model ANN (Keras Sequential):  
  Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
- Optimizer: Adam, Loss: Binary Crossentropy.
- EarlyStopping + Threshold tuning (by F1)
- Visualisasi: ROC, PR, Learning Curve.
- Output: `ann_p7.h5`, `scaler_p7.pkl`

---

## ⚙️ Cara Menjalankan
1. Pastikan file `processed_kelulusan.csv` tersedia.
2. Jalankan script sesuai pertemuan:
   ```bash
   python P6_random_forest.py
3. Semua grafik & model akan tersimpan otomatis.

## 💾 Requirements : 
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
joblib
