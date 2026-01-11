import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from pathlib import Path
import json

import sys


# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data/processed_data'
METRICS_PATH = PROJECT_ROOT / 'metrics'

# Charger les données
X_test = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')
y_test = pd.read_csv(DATA_DIR / 'y_test.csv')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_select = [col for col in X_test.columns if col != 'date']
X_test = X_test[cols_to_select]


# Load your saved model
loaded_model = joblib.load(MODEL_DIR / "final_model.joblib")

y_pred = loaded_model.predict(X_test)


# Évaluation complète
metrics = {
    'rmse': root_mean_squared_error(y_test, y_pred),  # ✅ Fix
    'mae': mean_absolute_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}

print(f"✅ RMSE: {metrics['rmse']:.4f}")
print(f"✅ MAE:  {metrics['mae']:.4f}")
print(f"✅ R²:   {metrics['r2']:.4f}")


# Sauvegarder les predictions
pd.DataFrame(y_pred).to_csv(PROJECT_ROOT / 'data/y_pred.csv', index=False)
print("✅ y_pred sauvé en CSV (pandas)")


# Sauvegarde (dans models/ avec ton modèle)
with open(METRICS_PATH / 'scores.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métriques sauvegardées: {METRICS_PATH}")

# Statistiques
print(f"Paramètres utilisés : {loaded_model.get_params()}")



