import joblib
from sklearn.ensemble import RandomForestRegressor
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
X_train = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
y_train = pd.read_csv(DATA_DIR / 'y_train.csv')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_select = [col for col in X_train.columns if col != 'date']
X_train = X_train[cols_to_select]

# Charger les meilleurs paramètres
best_params = joblib.load(PARAM_DIR / 'best_params_rf.pkl')

print(f"Paramètres chargés : {best_params}")


# Créer et entraîner le modèle final
final_model = RandomForestRegressor(**best_params, random_state=42)

# Entraînement + prédictions
final_model.fit(X_train, y_train)


#--Save the trained model to a file
joblib.dump(final_model, PARAM_DIR / 'final_model.joblib')
print("Modèle final sauvegardé dans models/final_model.joblib")

# Statistiques
print(f"Paramètres utilisés : {final_model.get_params()}")



