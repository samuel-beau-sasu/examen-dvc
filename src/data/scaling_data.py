# Normalisation des données
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Charge le dataset
X_train = pd.read_pickle('../data/processed_data/X_train.pkl')
#X_train = pd.read_csv('../data/processed_data/X_train.csv')
# 1. Charge le dataset
X_test = pd.read_pickle('../data/processed_data/X_test.pkl')
#X_test = pd.read_csv('../data/processed_data/X_test.csv')

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
X_train.to_pickle('../data/processed_data/X_train_scaled.pkl')
X_test.to_pickle('../data/processed_data/X_test_scaled.pkl')


# test
print("✅ Scaling sur toutes colonnes sauf 'date'")
print("Date inchangée:", X_train_scaled['date'].equals(X_train['date']))  # True

print("Train avant:", X_train[cols_to_scale].mean()[:3])  # ✅ Seulement numériques
print("Train après:", X_train_scaled[cols_to_scale].mean()[:3])  # ~0
print("Test après: ", X_test_scaled[cols_to_scale].mean()[:3])   # ~0