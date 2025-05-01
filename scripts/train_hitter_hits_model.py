# train_hitter_hits_model.py
# Trains a model to predict the number of hits a batter will get in a game.

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import time
import os
import traceback
import warnings
import pickle
from tqdm import tqdm
import gc
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split # Will use time-based split later
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None # default='warn'

# === Configuration & Constants ===
DB_PATH = "mlb_data2.db"
TABLE_NAME = "pitch_data"
LOG_FILE = "train_hitter_hits.log"
MODEL_OUTPUT_PATH = "hitter_hits_model.pkl"
FEATURES_OUTPUT_PATH = "hitter_hits_features.pkl"

# Date range for training/evaluation data
DATA_START_DATE = "2023-03-30"
DATA_END_DATE = "2023-10-01"
# Define the date to split training and testing data
# Example: Train on data up to Sep 1st, test on Sep 1st onwards
TEST_SET_START_DATE = "2023-09-01"

CHUNKSIZE = 20000 # Adjust based on memory

# Feature Engineering Parameters
ROLLING_GAMES = [15, 30] # Window sizes in terms of games played

# === Logging Setup ===
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# === Helper Functions ===
def safe_get_numeric(series, default=0.0):
    if series is None: return pd.Series(dtype=float)
    if series.empty: return series
    return pd.to_numeric(series, errors='coerce').fillna(default)

# === Core Logic Functions ===

def load_data_for_hits(start_date_str, end_date_str):
    """Loads pitch data, selecting columns needed for Hits prediction."""
    logging.info(f"Loading pitch data from {start_date_str} to {end_date_str} from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found: {DB_PATH}")
        return pd.DataFrame()

    # Columns needed for target (H), base stats (AB, PA, BB, SO), and features/context
    required_cols = [
        'game_date', 'game_pk', 'batter', 'events', 'description',
        'inning_topbot', 'home_team', 'away_team',
        'woba_denom', # To identify PA ending events reliably
        'bb_type', # Needed to help define AB (e.g., exclude walks, HBP)
        # Add more columns if needed for features later (e.g., pitcher ID, pitch types)
    ]

    db_cols_query = f"PRAGMA table_info({TABLE_NAME});"
    available_cols = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(db_cols_query)
            available_cols = [info[1] for info in cursor.fetchall()]
    except sqlite3.Error as e:
        logging.error(f"SQLite error getting columns: {e}")
        return pd.DataFrame()

    select_cols_sql = []
    actual_cols_loaded = []
    for col in required_cols:
        if col in available_cols:
            select_cols_sql.append(f'"{col}"')
            actual_cols_loaded.append(col)
        else:
             logging.warning(f"Required column '{col}' not found in database. It will be skipped.")

    essential_check = ['game_date', 'batter', 'events', 'game_pk']
    if not all(c in actual_cols_loaded for c in essential_check):
        missing = [c for c in essential_check if c not in actual_cols_loaded]
        logging.error(f"Missing essential columns for Hits target/features: {missing}. Cannot proceed.")
        return pd.DataFrame()

    select_cols_str = ', '.join(list(set(select_cols_sql)))
    query = f"SELECT {select_cols_str} FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
    params = (start_date_str, end_date_str)
    logging.debug(f"Executing query: {query} with params {params}")

    all_chunks = []
    # ... (Loading logic with tqdm as before) ...
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_rows = 0; n_chunks = None
            try:
                count_cursor = conn.cursor()
                count_query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
                count_cursor.execute(count_query, params)
                total_rows = count_cursor.fetchone()[0]
                count_cursor.close()
                if total_rows > 0: n_chunks = max(1, (total_rows // CHUNKSIZE) + (1 if total_rows % CHUNKSIZE > 0 else 0))
                else: n_chunks = 0
                logging.info(f"Estimated rows: {total_rows} in approx {n_chunks} chunks.")
            except Exception as e: logging.warning(f"Count estimate failed: {e}")

            if total_rows > 0:
                chunk_iterator = pd.read_sql_query(query, conn, params=params, chunksize=CHUNKSIZE)
                for chunk in tqdm(chunk_iterator, total=n_chunks, desc="Loading Data"):
                    if not chunk.empty:
                        chunk['game_date'] = pd.to_datetime(chunk['game_date'], errors='coerce')
                        chunk.dropna(subset=['game_date'], inplace=True)
                        all_chunks.append(chunk)
            else: logging.warning("Skipping loading: no rows estimated.")
    except Exception as e:
        logging.error(f"Error during data loading: {e}", exc_info=True)
        return pd.DataFrame()

    if not all_chunks: return pd.DataFrame()
    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Successfully loaded {len(df)} rows.")
    return df


def calculate_target_and_base_stats(df):
    """Aggregates pitch data to calculate actual Hits (target) and base stats per batter-game."""
    logging.info("Calculating target variable (Hits) and base stats...")
    if df.empty: return pd.DataFrame()

    # Ensure necessary columns exist
    required = ['game_date', 'game_pk', 'batter', 'events', 'woba_denom', 'bb_type']
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        logging.error(f"Missing required columns for aggregation: {missing}")
        return pd.DataFrame()

    # Convert IDs
    for id_col in ['batter', 'game_pk']:
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
    df.dropna(subset=['batter', 'game_pk', 'game_date'], inplace=True)
    df['batter'] = df['batter'].astype(int)
    df['game_pk'] = df['game_pk'].astype(int)

    # --- Calculate PA (Plate Appearance) ---
    if 'woba_denom' in df.columns and df['woba_denom'].notna().any():
        df['is_pa_outcome'] = (safe_get_numeric(df['woba_denom'], 0) == 1).astype(int)
    else:
        logging.warning("Missing 'woba_denom'. Inferring PA from events.")
        pa_ending_events = ['strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run','field_out', 'sac_fly', 'sac_bunt', 'force_out', 'grounded_into_double_play','fielders_choice', 'field_error', 'double_play', 'triple_play', 'batter_interference','catcher_interf']
        df['is_pa_outcome'] = df['events'].astype(str).isin(pa_ending_events).astype(int)

    # --- Event Flags for Base Stats ---
    event_flags = {'single': 'single', 'double': 'double', 'triple': 'triple', 'home_run': 'home_run',
                   'walk': 'walk', 'hit_by_pitch': 'hit_by_pitch', 'strikeout': 'strikeout',
                   'sac_fly': 'sac_fly', 'sac_bunt': 'sac_bunt'} # Add sac flies/bunts
    events_str = df['events'].astype(str)
    for event, flag_col in event_flags.items():
        if flag_col not in df.columns: df[flag_col] = 0
        df[flag_col] = (events_str == event).astype(int)
    # Refine SO from description if possible
    if 'description' in df.columns:
        desc = df['description'].astype(str).str.lower()
        if 'strikeout' not in df.columns: df['strikeout'] = 0
        df.loc[desc.str.contains('strikes out', na=False), 'strikeout'] = 1

    # --- Calculate AB (At Bat) ---
    df['is_walk'] = df['walk']
    df['is_hbp'] = df['hit_by_pitch']
    df['is_sh'] = df['sac_bunt']
    df['is_sf'] = df['sac_fly']
    df['is_ci'] = (events_str == 'catcher_interf').astype(int)
    df['non_ab_event'] = df['is_walk'] + df['is_hbp'] + df['is_sh'] + df['is_sf'] + df['is_ci']
    df['is_ab'] = ((df['is_pa_outcome'] == 1) & (df['non_ab_event'] == 0)).astype(int)

    # --- Calculate Hits (TARGET VARIABLE) ---
    df['H'] = df['single'] + df['double'] + df['triple'] + df['home_run']

    # --- Aggregation ---
    group_cols = ['game_date', 'game_pk', 'batter']
    agg_ops = {
        'H': 'sum',             # Target Variable
        'is_pa_outcome': 'sum', # Plate Appearances (PA)
        'is_ab': 'sum',         # At Bats (AB)
        'is_walk': 'sum',       # Walks (BB)
        'strikeout': 'sum',     # Strikeouts (SO)
        'single': 'sum',        # Needed for features if desired
        'double': 'sum',
        'triple': 'sum',
        'home_run': 'sum',
    }
    valid_agg_ops = {k: v for k, v in agg_ops.items() if k in df.columns}

    if 'H' not in valid_agg_ops:
        logging.error("Cannot calculate target variable 'H'. Check event flags.")
        return pd.DataFrame()

    daily_stats = df.groupby(group_cols, as_index=False, observed=True).agg(valid_agg_ops)
    daily_stats.rename(columns={'is_pa_outcome': 'PA', 'is_ab': 'AB', 'is_walk': 'BB', 'strikeout': 'SO'}, inplace=True)

    # Ensure integer types
    int_cols = ['H', 'PA', 'AB', 'BB', 'SO', 'single', 'double', 'triple', 'home_run']
    for col in int_cols:
        if col in daily_stats.columns:
             daily_stats[col] = daily_stats[col].astype(int)

    logging.info(f"Calculated target and base stats for {len(daily_stats)} batter-game instances.")
    return daily_stats


def create_hitter_features(df):
    """Creates rolling and lagged features for predicting hits."""
    logging.info("Creating features for Hitter Hits model...")
    if df.empty: return pd.DataFrame(), []
    if not all(c in df.columns for c in ['batter', 'game_date', 'AB', 'PA', 'BB', 'SO', 'H']):
        logging.error("Missing required columns for feature engineering.")
        return pd.DataFrame(), []

    df = df.sort_values(by=['batter', 'game_date', 'game_pk']).copy()

    # Calculate base rates needed for rolling averages
    # Use np.where to avoid division by zero safely
    df['AVG'] = np.where(df['AB'] > 0, df['H'] / df['AB'], 0.0)
    df['K_pct'] = np.where(df['PA'] > 0, df['SO'] / df['PA'], 0.0)
    df['BB_pct'] = np.where(df['PA'] > 0, df['BB'] / df['PA'], 0.0)


    # --- Calculate Rolling Features ---
    features = []
    grouped = df.groupby('batter')

    for G in ROLLING_GAMES:
        min_g = max(1, G // 3) # Min periods based on games

        # Rolling Sums (used to calculate rates over the window)
        # Use .shift(1) to prevent data leakage from the current game
        df[f'roll_{G}g_H_sum'] = grouped['H'].transform(lambda x: x.rolling(window=G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_AB_sum'] = grouped['AB'].transform(lambda x: x.rolling(window=G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_PA_sum'] = grouped['PA'].transform(lambda x: x.rolling(window=G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_BB_sum'] = grouped['BB'].transform(lambda x: x.rolling(window=G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_SO_sum'] = grouped['SO'].transform(lambda x: x.rolling(window=G, min_periods=min_g).sum().shift(1))

        # Rolling Rates (calculated from rolling sums)
        avg_col = f'roll_{G}g_AVG'
        k_pct_col = f'roll_{G}g_K_pct'
        bb_pct_col = f'roll_{G}g_BB_pct'

        # Calculate rates safely using np.where for division
        roll_ab = safe_get_numeric(df[f'roll_{G}g_AB_sum'])
        roll_pa = safe_get_numeric(df[f'roll_{G}g_PA_sum'])
        df[avg_col] = np.where(roll_ab > 0, safe_get_numeric(df[f'roll_{G}g_H_sum']) / roll_ab, 0.0)
        df[k_pct_col] = np.where(roll_pa > 0, safe_get_numeric(df[f'roll_{G}g_SO_sum']) / roll_pa, 0.0)
        df[bb_pct_col] = np.where(roll_pa > 0, safe_get_numeric(df[f'roll_{G}g_BB_sum']) / roll_pa, 0.0)

        features.extend([avg_col, k_pct_col, bb_pct_col])

    # --- Lagged Features ---
    df['lag_1g_H'] = grouped['H'].shift(1)
    features.append('lag_1g_H')

    # --- TODO: Add Park Factor, Opponent Features ---
    df['park_factor_H'] = 1.0 # Neutral park factor placeholder
    df['vs_LHP'] = 0 # Indicator placeholder
    df['vs_RHP'] = 0 # Indicator placeholder
    features.extend(['park_factor_H', 'vs_LHP', 'vs_RHP'])

    # Fill NaNs created by rolling/lagging (use 0 for simplicity)
    df[features] = df[features].fillna(0)

    logging.info(f"Created {len(features)} features.")
    return df, features


def train_and_evaluate(df, features, target_col='H'):
    """Splits data, trains XGBoost model, evaluates, and saves."""
    logging.info("Splitting data, training model, and evaluating...")
    if df.empty or not features:
        logging.error("No data or features available for training.")
        return

    # Time-based split
    df['game_date'] = pd.to_datetime(df['game_date'])
    train_df = df[df['game_date'] < TEST_SET_START_DATE].copy()
    test_df = df[df['game_date'] >= TEST_SET_START_DATE].copy()

    if train_df.empty or test_df.empty:
        logging.error("Train or test split resulted in empty DataFrame. Check date range and TEST_SET_START_DATE.")
        return

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # --- Model Training ---
    logging.info("Training XGBoost Regressor model...")
    model = XGBRegressor(
        objective='count:poisson', # Suitable for count data like hits
        n_estimators=100,         # Number of trees (tune later)
        learning_rate=0.1,        # Step size shrinkage (tune later)
        max_depth=3,              # Max depth of trees (tune later)
        subsample=0.8,            # Fraction of samples used per tree
        colsample_bytree=0.8,     # Fraction of features used per tree
        random_state=42,
        n_jobs=-1                 # Use all available CPU cores
        # Removed early_stopping_rounds from init
    )

    try:
        # Pass eval_set for potential internal use or logging, but not early stopping here
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  # early_stopping_rounds=10, # Removed this parameter
                  verbose=False) # Set verbose=True during tuning if needed
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        return

    logging.info("Model training complete.")

    # --- Evaluation ---
    predictions = model.predict(X_test)
    # Ensure predictions are non-negative before rounding
    predictions_non_negative = np.maximum(0, predictions)
    predictions_rounded = np.round(predictions_non_negative).astype(int)

    mae = mean_absolute_error(y_test, predictions_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_rounded))
    # R2 might be less informative for count data, but can include
    r2 = r2_score(y_test, predictions) # Use raw predictions for R2

    logging.info("--- Model Evaluation ---")
    logging.info(f"Target: {target_col}")
    logging.info(f"Test Set MAE:  {mae:.4f}")
    logging.info(f"Test Set RMSE: {rmse:.4f}")
    logging.info(f"Test Set R²:   {r2:.4f}")
    logging.info("------------------------")

    # --- Save Model and Features ---
    try:
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        with open(FEATURES_OUTPUT_PATH, 'wb') as f:
            pickle.dump(features, f)
        logging.info(f"Feature list saved to {FEATURES_OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Error saving model or features: {e}", exc_info=True)


# === Main Execution ===
if __name__ == "__main__":
    logging.info("--- Starting Hitter Hits Model Training ---")
    start_time = time.time()

    # 1. Load Data
    raw_data = load_data_for_hits(DATA_START_DATE, DATA_END_DATE)

    if not raw_data.empty:
        # 2. Calculate Target and Base Stats
        daily_stats_df = calculate_target_and_base_stats(raw_data)
        del raw_data; gc.collect()

        if not daily_stats_df.empty:
            # 3. Create Features
            features_df, feature_names = create_hitter_features(daily_stats_df)
            del daily_stats_df; gc.collect()

            if not features_df.empty and feature_names:
                # 4. Train and Evaluate
                train_and_evaluate(features_df, feature_names, target_col='H')
            else:
                logging.error("Feature creation failed.")
        else:
            logging.error("Target variable calculation failed.")
    else:
        logging.error("Data loading failed.")

    end_time = time.time()
    logging.info(f"--- Script Finished ---")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
