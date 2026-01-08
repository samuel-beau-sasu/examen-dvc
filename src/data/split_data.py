# Split des données en ensemble d'entraînement et de test.
# 4 datasets (X_test, X_train, y_test, y_train)

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Charge le CSV (téléchargé avec curl)
df = pd.read_csv('../data/raw_data/raw.csv')

# 2. Sépare features X et target y
X = df.drop('silica_concentrate', axis=1)  # Toutes les features
y = df['silica_concentrate']               # La cible

# 3. Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% test
    random_state=42,      # Reproductible
)

print(f"✅ Train: {X_train.shape}, {y_train.shape}")
print(f"✅ Test:  {X_test.shape}, {y_test.shape}")

# 4. Sauvegarde chaque dataset
#X_train.to_csv('../data/processed_data/X_train.csv', index=False)
#X_test.to_csv('../data/processed_data/X_test.csv', index=False)
#y_train.to_csv('../data/processed_data/y_train.csv', index=False)
#y_test.to_csv('../data/processed_data/y_test.csv', index=False)

# Sauvegarde en format dataset (.pkl)
X_train.to_pickle('../data/processed_data/X_train.pkl')
X_test.to_pickle('../data/processed_data/X_test.pkl')
y_train.to_pickle('../data/processed_data/y_train.pkl')
y_test.to_pickle('../data/processed_data/y_test.pkl')

print("✅ Datasets sauvegardés dans ../data/processed_data/")
print("📁 Fichiers:", os.listdir('../data/processed_data'))

