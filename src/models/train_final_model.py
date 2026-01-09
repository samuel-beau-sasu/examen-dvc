import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from pathlib import Path
import json



# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent

PARAM_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data/processed_data'
METRICS_PATH = PROJECT_ROOT / 'metrics'

# Charger les données
X_train = pd.read_pickle(DATA_DIR / 'X_train_scaled.pkl')
X_test = pd.read_pickle(DATA_DIR / 'X_test_scaled.pkl')
y_train = pd.read_pickle(DATA_DIR / 'y_train.pkl')
y_test = pd.read_pickle(DATA_DIR / 'y_test.pkl')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_select = [col for col in X_train.columns if col != 'date']
X_train = X_train[cols_to_select]
X_test = X_test[cols_to_select]

# Charger les meilleurs paramètres
#best_params = joblib.load('../models/best_params.pkl')
best_params = joblib.load(PARAM_DIR / 'best_params_sc.pkl')

print(f"Paramètres chargés : {best_params}")


# OPTION : Entraîner sur TOUTES les données (train + test)
# pour maximiser les performances en production
X_full = np.vstack([X_train, X_test])
y_full = np.concatenate([y_train, y_test])

# Créer et entraîner le modèle final
final_model = GradientBoostingRegressor(**best_params, random_state=42)

# Entraînement + prédictions
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Évaluation complète
metrics = {
    'rmse': root_mean_squared_error(y_test, y_pred),  # ✅ Fix
    'mae': mean_absolute_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}

print(f"✅ RMSE: {metrics['rmse']:.4f}")
print(f"✅ MAE:  {metrics['mae']:.4f}")
print(f"✅ R²:   {metrics['r2']:.4f}")

print(metrics)

# Sauvegarder le modèle
joblib.dump(final_model, PARAM_DIR / 'final_model.pkl')
print("✅ Modèle final sauvegardé dans models/final_model.pkl")

# Sauvegarde (dans models/ avec ton modèle)
with open(METRICS_PATH / 'scores.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métriques sauvegardées: {METRICS_PATH}")

# Statistiques
print(f"\nModèle entraîné sur {len(X_full)} échantillons")
print(f"Paramètres utilisés : {final_model.get_params()}")



