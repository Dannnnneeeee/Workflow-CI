"""
================================================================================
MODELLING CI - MLflow Project Compatible
================================================================================
File: modelling.py (for CI/CD with MLflow Project)
Author: Muhammad Wildan
Description: Training pipeline compatible with MLflow Project for CI/CD
================================================================================
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for Toyota price prediction')
    parser.add_argument('--data_path', type=str, default='toyota_clean.csv',
                       help='Path to preprocessed data')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=6,
                       help='Maximum tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    return parser.parse_args()

def load_and_prepare_data(data_path, test_size):
    """Load and prepare data for training"""
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} rows")
    
    # Features
    feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                       'model_encoded', 'transmission_encoded', 'fuelType_encoded']
    
    X = df[feature_columns]
    y = df['price']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, params):
    """Train XGBoost model"""
    print("\n[2/5] Training XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    print("✓ Training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n[3/5] Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    print(f"✓ RMSE: ${metrics['rmse']:,.2f}")
    print(f"✓ MAE: ${metrics['mae']:,.2f}")
    print(f"✓ R² Score: {metrics['r2']:.4f}")
    
    return metrics

def main():
    """Main training pipeline"""
    print("="*70)
    print("MLFLOW PROJECT - CI/CD TRAINING")
    print("="*70)
    
    # Parse arguments
    args = parse_args()
    
    print(f"\nParameters:")
    print(f"  • Data: {args.data_path}")
    print(f"  • n_estimators: {args.n_estimators}")
    print(f"  • max_depth: {args.max_depth}")
    print(f"  • learning_rate: {args.learning_rate}")
    print(f"  • test_size: {args.test_size}")
    
    # ========================================================================
    # FIX: Check if running inside MLflow Project
    # ========================================================================
    is_mlflow_project = os.getenv('MLFLOW_RUN_ID') is not None
    
    if is_mlflow_project:
        print("\n✓ Running inside MLflow Project (using existing run)")
        # Don't create new run, don't set experiment
        run_context = None
    else:
        print("\n✓ Running standalone (creating new run)")
        # Set experiment and create run only if standalone
        mlflow.set_experiment("Toyota_CI_Training")
        run_context = mlflow.start_run()
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            args.data_path, args.test_size
        )
        
        # Log parameters
        print("\n[4/5] Logging to MLflow...")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Train model
        params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate
        }
        model = train_model(X_train, y_train, params)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("rmse", metrics['rmse'])
        mlflow.log_metric("mae", metrics['mae'])
        mlflow.log_metric("r2", metrics['r2'])
        
        # Log model
        print("\n[5/5] Saving model...")
        mlflow.xgboost.log_model(model, "model")
        
        print("\n Pipeline completed successfully!")
        print("="*70)
        
        return metrics
        
    finally:
        # Only end run if we created it (standalone mode)
        if run_context is not None:
            mlflow.end_run()

if __name__ == "__main__":
    main()