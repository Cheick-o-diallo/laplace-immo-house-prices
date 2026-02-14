# 🏠 Laplace Immo ML - Simulateur Prix des Maisons

**Algorithme de prédiction des prix immobiliers** pour le réseau national d'agences Laplace Immo. Basé sur le dataset Ames Housing (79 features).

## 🏆 Performances Champion (Lasso)
| Métrique | Valeur |
|----------|--------|
| **Test RMSE** | **0.1219** |
| **Test R²** | **0.9203** |
| Résidus Moyenne | -0.0044 |
| Résidus Médiane | 0.0022 |
| Résidus Écart-type | 0.1221 |

## 🚀 Installation rapide

```bash
git clone <ton-repo>
cd laplace-immo-ml
pip install -r requirements.txt

📊 Utilisation
1. Lancer MLflow UI (Interface web)
bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
📱 Ouvre http://localhost:5000 pour voir tous les runs, comparer modèles et télécharger les meilleurs.
2. Entraîner le modèle champion
python src/mlflow_train_lasso.py
✅ Résultat : Run "Lasso_Champion" créé avec RMSE 0.1219.
3. Test unitaire
pytest tests/ --cov=src/
```
## 👥 Équipe Menbres
cheick O Diallo,
Dodzi Ahnert,
Issouf Bamba,
Abdoulaye Dioro Cissé.

## 📄 Licence
Propriété Laplace Immo © 2026. Usage interne uniquement.
