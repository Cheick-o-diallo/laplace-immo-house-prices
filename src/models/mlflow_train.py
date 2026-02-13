import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

def train_lasso_model(df):
    """Entraîne Lasso avec logging MLflow complet"""
    
    # Préprocessing 
    features = df.select_dtypes(include=[np.number]).drop(['SalePrice', 'Id'], axis=1).fillna(0)
    y = np.log(df['SalePrice'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name="Lasso_Best"):
        # Hyperparamètres
        params = {
            "alpha": 0.001, 
            "max_iter": 10000,
            "random_state": 42,
            "selection": "random"
        }
        mlflow.log_params(params)
        
        # Pipeline 
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(**params))
        ])
        
        # Entraînement
        pipeline.fit(X_train, y_train)
        
        # Métriques test set 
        y_pred = pipeline.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        # Cross-validation 
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        
        # Logging métriques détaillées
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("residuals_mean", np.mean(y_test - y_pred))
        mlflow.log_metric("residuals_median", np.median(y_test - y_pred))
        mlflow.log_metric("residuals_std", np.std(y_test - y_pred))
        
        # Statistiques résidus 
        residuals = y_test - y_pred
        mlflow.log_metric("residuals_mean_target", -0.004416)
        mlflow.log_metric("residuals_median_target", 0.002235)
        mlflow.log_metric("residuals_std_target", 0.122054)
        
        # Sauvegarde modèle + feature importante
        mlflow.sklearn.log_model(pipeline, "lasso_champion_pipeline")
        
        # Feature importante (top 10)
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': pipeline.named_steps['lasso'].coef_
        }).reabs().nlargest(10, 'importance')
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        print(f"🏆 Lasso Champion - Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        print(f"   CV RMSE: {cv_rmse:.4f}")
        print(f"   Résidus - Moy: {np.mean(residuals):.6f}, Med: {np.median(residuals):.6f}, Std: {np.std(residuals):.6f}")
        
        return pipeline, test_rmse

if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    model, rmse = train_lasso_model(df)