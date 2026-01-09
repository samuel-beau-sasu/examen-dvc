# récupération des données bruts 
cd /home/ubuntu/examen-dvc/data/raw_data

wget "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

# La commande uv init permet de créer un projet Python structuré

uv init --name dvc

uv add pandas
uv add scikit-learn

.venv/bin/activate

uv run python src/data/split_data.py
uv run python src/data/scaling_data.py

uv run python src/models/grid_search.py