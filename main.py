import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Baca dataset dari file CSV
dataset = pd.read_csv("Data.csv")

# Menggunakan LabelEncoder untuk mengubah kolom target menjadi nilai numerik
le = LabelEncoder()
sc = StandardScaler()
y = le.fit_transform(dataset.iloc[:, -1].values)

# Memilih fitur-fitur (kolom-kolom) yang akan digunakan
X = dataset.iloc[:, :-1].values

# Menggunakan SimpleImputer untuk mengisi missing values dengan mean pada kolom-kolom yang membutuhkan
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Menggunakan OneHotEncoder untuk mengkodekan kolom kategori
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Menggunakan StandardScaler untuk menskalakan fitur-fitur numerik
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# Cetak data X uji (fitur-fitur) yang telah diproses
print(X_test)