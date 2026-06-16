# train_model.py
# Mereplikasi pipeline dari Tubes_Kelompok2.ipynb untuk menghasilkan
# best_twitch_partner_model.joblib dan twitch_scaler.joblib

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

URL = "https://raw.githubusercontent.com/DachsteinSilalahi/Dataset_kelompok2/refs/heads/main/twitchdata-update.csv"

print("Membaca dataset...")
df = pd.read_csv(URL)

# 31. Penentuan variabel target dan fitur
X = df.drop(columns=['Channel', 'Partnered'])
y = df['Partnered']

# 32. Encoding variabel kategorikal
X_encoded = pd.get_dummies(X, columns=['Language', 'Mature'], drop_first=True)
feature_columns = X_encoded.columns.tolist()

# 33. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 34. Standardisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 37. SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 42. Hyperparameter tuning RandomForest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

print("Menjalankan GridSearchCV...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_res, y_train_res)

print("Parameter terbaik:", grid_search.best_params_)
print("Best F1-Score (CV):", grid_search.best_score_)

best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test_scaled)

print("\nClassification report (test set):")
print(classification_report(y_test, y_pred_best))
print("Akurasi test:", accuracy_score(y_test, y_pred_best))

# 45. Simpan model, scaler, dan daftar kolom fitur
joblib.dump(best_rf_model, 'best_twitch_partner_model.joblib')
joblib.dump(scaler, 'twitch_scaler.joblib')
joblib.dump(feature_columns, 'feature_columns.joblib')

print("\nModel, scaler, dan daftar kolom fitur berhasil disimpan.")
