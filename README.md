# 🎓 Prediksi Kelulusan Mahasiswa — Machine Learning Project (Unpam)
Proyek ini berisi serangkaian latihan praktikum mata kuliah Machine Learning di Universitas Pamulang (UNPAM), berfokus pada pembuatan model prediksi kelulusan mahasiswa berdasarkan data akademik seperti IPK, kehadiran, dan waktu belajar.
Setiap pertemuan merepresentasikan tahapan berbeda dalam pipeline Machine Learning — dari data preparation, modeling, hingga neural network.

# 🧩 Struktur Folder
# 📂 ML_Kelulusan/
├── kelulusan_mahasiswa.csv        # dataset mentah
├── processed_kelulusan.csv        # hasil pembersihan + feature engineering
│
├── 📄 P4_data_preparation.py      # Pertemuan 4
├── 📄 P5_modeling.py              # Pertemuan 5
├── 📄 P6_random_forest.py         # Pertemuan 6
├── 📄 P7_ann.py                   # Pertemuan 7
│
├── cm_test_p5.png                 # confusion matrix (P5)
├── roc_test_p5.png                # ROC curve (P5)
├── cm_test_p6.png                 # confusion matrix (P6)
├── roc_test_p6.png                # ROC curve (P6)
├── pr_test_p6.png                 # precision-recall (P6)
├── learning_curve_p7.png          # loss curve (P7)
├── roc_ann_p7.png                 # ROC curve (P7)
├── pr_ann_p7.png                  # precision-recall (P7)
│
├── rf_model.pkl                   # model RandomForest tersimpan
├── ann_p7.h5                      # model ANN tersimpan
├── scaler_p7.pkl                  # scaler untuk preprocessing ANN
└── README.md

# 🧠 Tahapan Tiap Pertemuan
# 📘 Pertemuan 4 — Data Preparation
- Membersihkan data dari missing value & duplikasi.
- Eksplorasi data (EDA): boxplot, histogram, scatter plot, heatmap korelasi.
- Feature engineering:
  - Rasio_Absensi = Jumlah_Absensi / 14
  - IPK_x_Study = IPK * Waktu_Belajar_Jam
- Split data → 70 % train | 15 % val | 15 % test
- Hasil: processed_kelulusan.csv

📗 Pertemuan 5 — Modeling (Baseline vs Random Forest)
    - Baseline: Logistic Regression dengan pipeline preprocessing.
    - Model alternatif: Random Forest.
    - Evaluasi: F1, Precision, Recall, ROC-AUC, Confusion Matrix.
    - Tuning hyperparameter dengan GridSearchCV.
    - Hasil terbaik disimpan → model_p5.pkl.

📙 Pertemuan 6 — Random Forest (Optimasi Final)
    - Validasi silang (StratifiedKFold).
    - GridSearch tuning (max_depth, min_samples_split).
    - Visualisasi hasil: ROC, PR, Confusion Matrix.
    - Analisis Feature Importance → tiga fitur paling berpengaruh:
      IPK_x_Study, IPK, Rasio_Absensi.
    - Model final disimpan → rf_model.pkl.

📘 Pertemuan 7 — Artificial Neural Network (ANN)
    - Model ANN sederhana (Keras Sequential):
      - Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
    - Optimizer Adam, loss binary crossentropy, metric AUC.
    - Early Stopping + threshold tuning berdasarkan F1.
    - Hasil visualisasi: Learning Curve, ROC, PR Curve.
    - Model dan scaler disimpan → ann_p7.h5, scaler_p7.pkl.

⚙️ Requirement Package
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib

🚀 Cara Menjalankan
  1. Pastikan file processed_kelulusan.csv ada di folder.
2. Jalankan script per pertemuan, misalnya:

python P6_random_forest.py

  3. Semua grafik & model akan tersimpan otomatis di direktori kerja.

🧾 Output Utama
File	                              Deskripsi
processed_kelulusan.csv	            Dataset hasil pembersihan
cm_test_p6.png, roc_test_p6.png	    Hasil evaluasi Random Forest
learning_curve_p7.png              	Grafik loss ANN
rf_model.pkl, ann_p7.h5	            Model siap pakai
scaler_p7.pkl	                      Scaler untuk inference ANN

💬 Catatan
    - Gunakan seed = 42 agar hasil replikasi konsisten.
    - Semua pipeline dirancang agar tidak terjadi data leakage.
    - Model ANN dan Random Forest sudah diuji dengan inference lokal (contoh input mahasiswa).

Semua pipeline dirancang agar tidak terjadi data leakage.

Model ANN dan Random Forest sudah diuji dengan inference lokal (contoh input mahasiswa).
