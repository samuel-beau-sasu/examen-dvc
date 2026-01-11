import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path


# Chemin absolu depuis le script actuel
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data/processed_data'
PARAM_DIR = PROJECT_ROOT / 'models'

# Charger les données
X_train = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
X_test = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')
y_train = pd.read_csv(DATA_DIR / 'y_train.csv')
y_test = pd.read_csv(DATA_DIR / 'y_test.csv')


# Sélectionne colonnes à scaler (exclut 'date')
cols_to_select = [col for col in X_train.columns if col != 'date']
X_train = X_train[cols_to_select]
X_test = X_test[cols_to_select]


# Modèle et paramètres
# Modèle et paramètres pour RandomForest
model = RandomForestRegressor(random_state=42, n_jobs=1)

param_grid = {
    'n_estimators': [100, 200], # [100, 200, 300],      # Nombre d'arbres
    'max_depth': [None, 10, 20], # [None, 10, 20, 30],      # None = pas de limite (différent de GradientBoosting)
    'min_samples_split': [2, 5], # [2, 5, 10],      # Minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': [1, 2], # [1, 2, 4],        # Minimum d'échantillons dans une feuille
    'max_features': ['sqrt', 'log2'], # ['auto', 'sqrt', 'log2', 0.33, 0.5],  # Nombre de features à considérer
    'bootstrap': [True] # [True, False],           # Échantillonnage avec/sans remise
    #'max_samples': [None, 0.8]           # Proportion d'échantillons pour chaque arbre
}

# GridSearch
print("Recherche des meilleurs hyperparamètres (RandomForest)...")
grid_search = GridSearchCV(
    model, 
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,              # Attention: RandomForest utilise déjà n_jobs
    verbose=2,
    refit=True,             # Ré-entraîne sur X_train complet
    return_train_score=True
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
joblib.dump(grid_search.best_params_, PARAM_DIR / 'best_params_rf.pkl')


chemin_complet = PARAM_DIR / 'best_params_rf.pkl'
print(f"\n✅ Paramètres sauvegardés dans {chemin_complet}")