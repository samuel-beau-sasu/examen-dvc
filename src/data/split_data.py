# Split des données en ensemble d'entraînement et de test.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data/raw_data'
DATA_PRO = PROJECT_ROOT / 'data/processed_data'

DATA_PRO.mkdir(parents=True, exist_ok=True) 

# Charger les données
df = pd.read_csv(DATA_DIR / 'raw.csv')

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
X_train.to_csv(DATA_PRO / 'X_train.csv', index=False)
X_test.to_csv(DATA_PRO / 'X_test.csv', index=False)
y_train.to_csv(DATA_PRO / 'y_train.csv', index=False)
y_test.to_csv(DATA_PRO / 'y_test.csv', index=False)

print(f"✅ Datasets sauvegardés dans {DATA_PRO}")




