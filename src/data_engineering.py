import pandas as pd
import numpy as np

def transform_and_engineer_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw options chain into spatial-temporal variance grid."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
    
    contract_cols = [col for col in df.columns if col.startswith('NIFTY')]
    df_long = df.melt(id_vars=['datetime', 'underlying_price'], value_vars=contract_cols, var_name='contract', value_name='iv')
    
    regex = r'^NIFTY(?P<expiry_str>\d{2}[A-Z]{3}\d{2})(?P<strike>\d+)(?P<option_type>[C|P]E)$'
    extracted = df_long['contract'].str.extract(regex)
    df_long['strike'] = extracted['strike'].astype(float)
    df_long['option_type'] = extracted['option_type']
    
    df_long['expiry'] = pd.to_datetime(extracted['expiry_str'], format='%d%b%y') + pd.Timedelta(hours=15, minutes=30)
    df_long['log_moneyness'] = np.log(df_long['underlying_price'] / df_long['strike'])
    
    delta_seconds = (df_long['expiry'] - df_long['datetime']).dt.total_seconds()
    df_long['tau'] = (delta_seconds / (365.25 * 24 * 3600)).clip(lower=1e-5)
    
    df_long['hour'] = df_long['datetime'].dt.hour
    df_long['minute'] = df_long['datetime'].dt.minute
    df_long['time_to_close'] = (15 - df_long['hour']) * 60 + (30 - df_long['minute'])
    df_long['day_of_week'] = df_long['datetime'].dt.dayofweek
    
    # Domain Shift
    df_long['w'] = (df_long['iv'] ** 2) * df_long['tau']
    
    # Temporal Memory (Backward Time)
    df_long = df_long.sort_values(['datetime', 'strike', 'option_type']).reset_index(drop=True)
    groupby_temporal = df_long.groupby(['strike', 'option_type'])['w']
    df_long['w_lag_1'] = groupby_temporal.shift(1)
    df_long['w_lag_3'] = groupby_temporal.shift(3)
    df_long['w_lag_6'] = groupby_temporal.shift(6)
    
    # Spatial Windowing (Cross-Strike Constraints)
    df_long = df_long.sort_values(['datetime', 'option_type', 'strike']).reset_index(drop=True)
    groupby_spatial = df_long.groupby(['datetime', 'option_type'])['w']
    df_long['w_strike_up'] = groupby_spatial.shift(-1)
    df_long['w_strike_down'] = groupby_spatial.shift(1)
    
    return df_long

def handle_nans_and_split(df: pd.DataFrame) -> tuple:
    """Isolates inference targets and imputes illiquid lag features."""
    df = df.copy()
    df['is_missing_target'] = df['iv'].isna()
    
    lag_cols = ['w_lag_1', 'w_lag_3', 'w_lag_6']
    df[lag_cols] = df.groupby(['strike', 'option_type'])[lag_cols].bfill().ffill()
    timestamp_medians = df.groupby('datetime')[lag_cols].transform('median')
    df[lag_cols] = df[lag_cols].fillna(timestamp_medians)
    
    return df[~df['is_missing_target']].copy(), df[df['is_missing_target']].copy()

def create_purged_time_splits(df: pd.DataFrame, train_days=5, purge_days=1, valid_days=2) -> list:
    """Enforces Purged Group Time Series Split to prevent look-ahead bias."""
    dates = pd.to_datetime(df['datetime'].dt.date)
    unique_dates = np.sort(dates.unique())
    total_window = train_days + purge_days + valid_days
    
    splits = []
    for i in range(len(unique_dates) - total_window + 1):
        train_dates = unique_dates[i : i + train_days]
        valid_dates = unique_dates[i + train_days + purge_days : i + total_window]
        
        train_idx = df.index[dates.isin(train_dates)]
        valid_idx = df.index[dates.isin(valid_dates)]
        splits.append((train_idx, valid_idx))
    return splits