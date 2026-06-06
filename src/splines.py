import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

def fit_variance_baseline_and_derivatives(train_df: pd.DataFrame, inference_df: pd.DataFrame, smoothing=0.0001) -> tuple:
    """
    Fits Smoothing Spline on Total Variance, clamps wings via flat extrapolation, 
    and extracts local analytical derivatives.
    """
    train_out, inf_out = train_df.copy(), inference_df.copy()
    for df in [train_out, inf_out]:
        df['w_baseline'], df['dw_dk'], df['d2w_dk2'] = np.nan, np.nan, np.nan
    
    for dt, group in train_out.groupby('datetime'):
        unique_group = group.groupby('log_moneyness', as_index=False)['w'].mean()
        if len(unique_group) < 4: continue
            
        x, y_w = unique_group['log_moneyness'].values, unique_group['w'].values
        spline = UnivariateSpline(x, y_w, k=3, s=smoothing)
        spline_d1, spline_d2 = spline.derivative(n=1), spline.derivative(n=2)
        
        # Train Output
        train_mask = train_out['datetime'] == dt
        x_train = train_out.loc[train_mask, 'log_moneyness']
        train_out.loc[train_mask, 'w_baseline'] = spline(x_train)
        train_out.loc[train_mask, 'dw_dk'] = spline_d1(x_train)
        train_out.loc[train_mask, 'd2w_dk2'] = spline_d2(x_train)
        
        # Inference Output (Strict Flat Boundary Extrapolation)
        inf_mask = inf_out['datetime'] == dt
        if not inf_mask.any(): continue
        
        inf_x = inf_out.loc[inf_mask, 'log_moneyness'].values
        inf_y, inf_d1, inf_d2 = np.zeros_like(inf_x), np.zeros_like(inf_x), np.zeros_like(inf_x)
        
        for i, val in enumerate(inf_x):
            if val < x.min():
                inf_y[i], inf_d1[i], inf_d2[i] = y_w[0], 0, 0
            elif val > x.max():
                inf_y[i], inf_d1[i], inf_d2[i] = y_w[-1], 0, 0
            else:
                inf_y[i], inf_d1[i], inf_d2[i] = spline(val), spline_d1(val), spline_d2(val)
                
        inf_out.loc[inf_mask, 'w_baseline'] = inf_y
        inf_out.loc[inf_mask, 'dw_dk'] = inf_d1
        inf_out.loc[inf_mask, 'd2w_dk2'] = inf_d2

    return train_out, inf_out