# Normalisation des données
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed_data'

# Charger les données
X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
X_test = pd.read_csv(DATA_DIR / 'X_test.csv')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_scale = [col for col in X_train.columns if col != 'date']
X_train_numeric = X_train[cols_to_scale]
X_test_numeric = X_test[cols_to_scale]

# Scaling
scaler = StandardScaler()
X_train_scaled_num = scaler.fit_transform(X_train_numeric)
X_test_scaled_num = scaler.transform(X_test_numeric)

# Recombine avec colonne date (inchangée)
# Ajoute la colonne date directement
X_train_scaled = pd.DataFrame(X_train_scaled_num, columns=cols_to_scale, index=X_train.index)
X_train_scaled['date'] = X_train['date']

X_test_scaled = pd.DataFrame(X_test_scaled_num, columns=cols_to_scale, index=X_test.index)
X_test_scaled['date'] = X_test['date']


# Sauvegarde en format dataset (.pkl)
X_train_scaled.to_csv(DATA_DIR / 'X_train_scaled.csv')
X_test_scaled.to_csv(DATA_DIR / 'X_test_scaled.csv')


# test
print("✅ Scaling sur toutes colonnes sauf 'date'")
print("Date inchangée:", X_train_scaled['date'].equals(X_train['date']))  # True

print("Train avant:", X_train[cols_to_scale].mean()[:3])  # ✅ Seulement numériques
print("Train après:", X_train_scaled[cols_to_scale].mean()[:3])  # ~0
print("Test après: ", X_test_scaled[cols_to_scale].mean()[:3])   # ~0