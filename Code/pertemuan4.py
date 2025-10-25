  # DATA PREPARATION - PERTEMUAN 4
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 0) load
assert os.path.exists("kelulusan_mahasiswa.csv"), "File 'kelulusan_mahasiswa.csv' belum ada."
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())

# 1) cleaning
print(df.isnull().sum())
df = df.drop_duplicates()

# 2) EDA
plt.figure(); sns.boxplot(x=df['IPK']); plt.title("Boxplot IPK")
plt.tight_layout(); plt.savefig("p4_boxplot_ipk.png", dpi=120); plt.close()

print(df.describe())

plt.figure(); sns.histplot(df['IPK'], bins=10, kde=True); plt.title("Distribusi IPK")
plt.tight_layout(); plt.savefig("p4_hist_ipk.png", dpi=120); plt.close()

plt.figure(); sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title("IPK vs Waktu Belajar (Hue=Lulus)")
plt.tight_layout(); plt.savefig("p4_scatter_ipk_study.png", dpi=120); plt.close()

plt.figure(); sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.tight_layout(); plt.savefig("p4_heatmap_corr.png", dpi=120); plt.close()

# 3) feature engineering + save processed
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study']   = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)
print("processed_kelulusan.csv tersimpan")

# 4) splitting 70/15/15
X = df.drop('Lulus', axis=1)
y = df['Lulus']

# stratify SEKALI
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# split kedua TANPA stratify 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Train:\n", y_train.value_counts())
print("Val:\n",   y_val.value_counts())
print("Test:\n",  y_test.value_counts())

# (opsional) simpan split untuk P5/P6 Pilihan B
X_train.to_csv("X_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Split CSV tersimpan.")
