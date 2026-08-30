# Telco Customer Churn Prediction

Application FastAPI et Gradio pour entraîner puis servir un modèle de prédiction du churn client.

## Démarrage

Les dépendances sont gérées avec `uv` :

```powershell
uv sync --all-groups
```

Entraînez le modèle (cela crée `artifacts/model.json` et les métadonnées utilisées par l'API) :

```powershell
uv run python scripts/run_pipeline.py --input data/raw/chum.csv
```

Lancez ensuite l'application :

```powershell
uv run uvicorn src.app.main:app --reload
```

L'API est disponible sur `http://127.0.0.1:8000/docs` et l'interface Gradio sur `http://127.0.0.1:8000/ui`.
