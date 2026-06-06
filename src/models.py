import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 1

def train_damped_xgboost_spatial(train_df: pd.DataFrame, cv_splits: list) -> list:
    """Trains spatial XGBoost to correct localized microstructure noise."""
    features = [
        'log_moneyness', 'tau', 
        'w_lag_1', 'w_lag_3', 'w_lag_6',     
        'w_strike_up', 'w_strike_down',      
        'dw_dk', 'd2w_dk2',                  
        'time_to_close', 'day_of_week'       
    ]
    target = 'w_residual'
    
    train_df[target] = train_df['w'] - train_df['w_baseline']
    oof_xgb = pd.Series(np.nan, index=train_df.index)
    models = []
    
    xgb_params = {
        'objective': 'reg:squarederror', 
        'max_depth': 4, 
        'learning_rate': 0.02, 
        'subsample': 0.7, 
        'colsample_bytree': 0.8,
        'tree_method': 'hist', 
        'random_state': RANDOM_SEED
    }
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, y_train = train_df.loc[train_idx, features], train_df.loc[train_idx, target]
        X_valid, y_valid = train_df.loc[val_idx, features], train_df.loc[val_idx, target]
        
        model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=600)
        model_xgb.fit(X_train, y_train, verbose=False)
        oof_xgb.loc[val_idx] = model_xgb.predict(X_valid)
        models.append(model_xgb)
        
    valid_mask = ~oof_xgb.isna()
    
    damping_threshold = 0.000228
    train_df['damping_multiplier'] = (train_df['tau'] / damping_threshold).clip(upper=1.0)
    train_df['damped_w_residual'] = oof_xgb * train_df['damping_multiplier']
    
    train_df['oof_w_pred'] = (train_df['w_baseline'] + train_df['damped_w_residual']).clip(lower=1e-6)
    train_df['oof_iv_pred'] = np.sqrt(train_df['oof_w_pred'] / train_df['tau'])
    
    final_iv_rmse = np.sqrt(mean_squared_error(train_df.loc[valid_mask, 'iv'], train_df.loc[valid_mask, 'oof_iv_pred']))
    print(f"GLOBAL OOF DAMPED XGBoost IV RMSE: {final_iv_rmse:.7f}")
    
    return models

def generate_final_submission_spatial(inference_df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Generates predictions across PGTS folds and applies boundary damping."""
    df = inference_df.copy()
    features = [
        'log_moneyness', 'tau', 
        'w_lag_1', 'w_lag_3', 'w_lag_6',     
        'w_strike_up', 'w_strike_down',      
        'dw_dk', 'd2w_dk2',                  
        'time_to_close', 'day_of_week'       
    ]
    X_infer = df[features]
    
    n_rows = len(df)
    n_folds = len(models)
    xgb_preds = np.zeros(n_rows)
    
    for model in models:
        xgb_preds += model.predict(X_infer) / n_folds
        
    damping_threshold = 0.000228
    df['damping_multiplier'] = (df['tau'] / damping_threshold).clip(upper=1.0)
    df['damped_residual'] = xgb_preds * df['damping_multiplier']
    
    df['final_w'] = (df['w_baseline'] + df['damped_residual']).clip(lower=1e-6)
    df['final_predicted_iv'] = np.sqrt(df['final_w'] / df['tau'])
    
    dt_str = df['datetime'].dt.strftime('%d-%m-%Y %H:%M')
    df['id'] = dt_str + '||' + df['contract']
    return df[['id', 'final_predicted_iv']].rename(columns={'final_predicted_iv': 'value'})