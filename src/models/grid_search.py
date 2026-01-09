import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path


# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed_data'
PARAM_DIR = PROJECT_ROOT / 'models'

# Charger les données
X_train = pd.read_pickle(DATA_DIR / 'X_train_scaled.pkl')
X_test = pd.read_pickle(DATA_DIR / 'X_test_scaled.pkl')
y_train = pd.read_pickle(DATA_DIR / 'y_train.pkl')
y_test = pd.read_pickle(DATA_DIR / 'y_test.pkl')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_select = [col for col in X_train.columns if col != 'date']
X_train = X_train[cols_to_select]
X_test = X_test[cols_to_select]


# Modèle et paramètres
model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 1.0]
}


# GridSearch
print("Recherche des meilleurs hyperparamètres...")
grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    refit=True  # Ré-entraîne sur X_train complet
)

grid_search.fit(X_train, y_train)

# Résultats
print(f"\n✅ Meilleurs paramètres : {grid_search.best_params_}")
print(f"Meilleur score CV : {-grid_search.best_score_:.4f}")

# Évaluation rapide sur test
y_pred = grid_search.best_estimator_.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Score test - MSE : {test_mse:.4f}, R² : {test_r2:.4f}")

# Sauvegarder les paramètres
#joblib.dump(grid_search.best_params_, 'models/best_params_sc.pkl')
joblib.dump(grid_search.best_params_, PARAM_DIR / 'best_params_sc.pkl')

print("\n✅ Paramètres sauvegardés dans ../models/best_params_sc.pkl")