# récupération des données bruts 
cd /home/ubuntu/examen-dvc/data/raw_data

wget "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

# La commande uv init permet de créer un projet Python structuré

uv init --name dvc

uv add pandas
uv add scikit-learn
uv add git-filter-repo
uv add dvc


source .venv/bin/activate

uv run python src/data/split_data.py
uv run python src/data/scaling_data.py

uv run python src/models/grid_search.py
uv run python src/models/train_final_model.py

#
git remote add origin https://github.com/samuel-beau-sasu/examen-dvc.git


# 
git filter-repo --invert-paths --force \
  --path models/data/best_params.pkl \
  --path models/data/best_params_rf.pkl \
  --path models/data/best_params_rm.pkl \
  --path models/data/best_params_sc.pkl \
  --path models/data/best_params_sc2.pkl \
  --path models/data/final_model.pkl

git filter-repo --invert-paths \
  --path models/models/final_model.pkl


#

git add data/processed_data.dvc data/.gitignore

# retrouver les parametres
dvc add models/best_params_rf.pkl

git add models/best_params_rf.pkl.dvc
git commit -m "Recherche des parametres du model"

# création du model
dvc add models/final_model.joblib

git add models/final_model.joblib.dvc
git commit -m "Recherche des parametres du model"

# evaluation du model
dvc add data/y_pred.csv

git add metrics/scores.json data/y_pred.csv.dvc
git commit -m "evaluated a first Random Forest Classifier"

# Pipeline

# Supprime le tracking manuel et passe au pipeline (recommandé) :
dvc remove data/processed_data.dvc

dvc stage add -n split --force \
  -d src/data/split_data.py \
  -d data/raw_data \
  -o data/processed_data \
  python src/data/split_data.py

dvc stage add -n split --force \
  -d src/data/split_data.py \
  -d data/raw_data \
  -o data/processed_data/X_train.csv \
  -o data/processed_data/X_test.csv \
  -o data/processed_data/y_train.csv \
  -o data/processed_data/y_test.csv \
  python src/data/split_data.py

dvc repro

dvc stage add -n scale --force \
  -d src/data/scaling_data.py \
  -d data/processed_data \
  -o data/processed_data/X_train_scaled.csv \
  -o data/processed_data/X_test_scaled.csv \
  python src/data/scaling_data.py

dvc stage add -n scale --force \
  -d src/data/scaling_data.py \
  -d data/processed_data/X_train.csv \
  -d data/processed_data/X_test.csv \
  -o data/processed_data/X_train_scaled.csv \
  -o data/processed_data/X_test_scaled.csv \
  python src/data/scaling_data.py

dvc repro

dvc remove models/best_params_rf.pkl.dvc

dvc stage add -n params \
              -d src/models/grid_search.py \
              -d data/processed_data \
              -o models/best_params_rf.pkl \
              python src/models/grid_search.py

dvc repro

dvc remove models/final_model.joblib.dvc

dvc stage add -n train --force \
              -d src/models/train_final_model.py \
              -d data/processed_data \
              -d models/best_params_rf.pkl \
              -o models/final_model.joblib \
              python src/models/train_final_model.py

dvc repro

dvc remove data/y_pred.csv.dvc

dvc stage add -n evaluation --force \
              -d src/models/evaluation_model.py \
              -d data/processed_data \
              -d models/final_model.joblib \
              -o data/y_pred.csv \
              -M metrics/scores.json \
              python src/models/evaluation_model.py
              
dvc repro