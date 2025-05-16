# train_pitcher_so_model.py - v1
# Trains a model to predict pitcher strikeouts, incorporating pitch mix features.

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
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None # default='warn'

# === Configuration & Constants ===
DB_PATH = "mlb_data2.db"
TABLE_NAME = "pitch_data"
LOG_FILE = "train_pitcher_so.log"
MODEL_OUTPUT_PATH = "pitcher_so_model_v1.pkl"
FEATURES_OUTPUT_PATH = "pitcher_so_features_v1.pkl"

DATA_START_DATE = "2023-03-30"
DATA_END_DATE = "2023-10-01"
TEST_SET_START_DATE = "2023-09-01"

CHUNKSIZE = 20000

# Feature Engineering Parameters
ROLLING_GAMES_BASE = [5, 20] # Windows for K/9, BB/9
ROLLING_GAMES_PITCHMIX = [5, 20] # Windows for pitch mix features
NEW_PITCH_BASELINE_GAMES = 20 # How many previous games to check for "new" pitch

# Pitch Type Classification (Example - refine based on your data's pitch_type codes)
PITCH_CLASSIFICATION = {
    # Fastballs
    'FF': 'FASTBALL', 'FA': 'FASTBALL', # 4-Seam
    'SI': 'FASTBALL', 'FT': 'FASTBALL', # Sinker/2-Seam
    'FC': 'FASTBALL', # Cutter
    # Breaking Balls
    'SL': 'BREAKING', # Slider
    'CU': 'BREAKING', 'KC': 'BREAKING', # Curveball, Knuckle-Curve ('CU' is often Curveball)
    'SV': 'BREAKING', # Slurve? Check data
    'CS': 'BREAKING', # Slow Curve? Check data
    # Off-Speed
    'CH': 'OFFSPEED', # Changeup
    'FS': 'OFFSPEED', 'SP': 'OFFSPEED', # Splitter
    'KN': 'OFFSPEED', # Knuckleball
    # Unknown/Other
    'PO': 'OTHER', 'UN': 'OTHER', 'XX': 'OTHER', 'SC': 'OTHER', # Pitch Out, Unknown, etc.
    'EP': 'OTHER' # Eephus?
}


# === Logging Setup ===
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Set higher level for debugging pandas issues if needed
# logging.getLogger('pandas').setLevel(logging.DEBUG)
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

def load_data_for_pitcher_so(start_date_str, end_date_str):
    """Loads pitch data, selecting columns needed for Pitcher SO prediction."""
    logging.info(f"Loading pitch data from {start_date_str} to {end_date_str} from {DB_PATH}")
    if not os.path.exists(DB_PATH): return pd.DataFrame()

    required_cols = [
        'game_date', 'game_pk', 'pitcher', 'events', 'pitch_type',
        'description', 'inning_topbot', # Needed for Outs calculation
        # Add batter ID if calculating opponent K% later
    ]

    db_cols_query = f"PRAGMA table_info({TABLE_NAME});"
    available_cols = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(db_cols_query)
            available_cols = [info[1] for info in cursor.fetchall()]
    except sqlite3.Error as e: return pd.DataFrame()

    select_cols_sql = []
    actual_cols_loaded = []
    for col in required_cols:
        if col in available_cols:
            select_cols_sql.append(f'"{col}"')
            actual_cols_loaded.append(col)
        else: logging.warning(f"Required column '{col}' not found in database.")

    essential_check = ['game_date', 'pitcher', 'events', 'game_pk', 'pitch_type']
    if not all(c in actual_cols_loaded for c in essential_check):
        missing = [c for c in essential_check if c not in actual_cols_loaded]
        logging.error(f"Missing essential columns for SO target/features: {missing}. Cannot proceed.")
        return pd.DataFrame()

    select_cols_str = ', '.join(list(set(select_cols_sql)))
    query = f"SELECT {select_cols_str} FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
    params = (start_date_str, end_date_str)

    all_chunks = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            total_rows = 0; n_chunks = None
            try:
                count_cursor = conn.cursor()
                count_query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
                count_cursor.execute(count_query, params)
                total_rows = count_cursor.fetchone()[0]; count_cursor.close()
                if total_rows > 0: n_chunks = max(1, (total_rows // CHUNKSIZE) + (1 if total_rows % CHUNKSIZE > 0 else 0))
                else: n_chunks = 0
                logging.info(f"Estimated rows: {total_rows} in approx {n_chunks} chunks.")
            except Exception as e: logging.warning(f"Count estimate failed: {e}")

            if total_rows > 0:
                chunk_iterator = pd.read_sql_query(query, conn, params=params, chunksize=CHUNKSIZE)
                for chunk in tqdm(chunk_iterator, total=n_chunks, desc="Loading Data"):
                    if not chunk.empty:
                        chunk['game_date'] = pd.to_datetime(chunk['game_date'], errors='coerce')
                        chunk.dropna(subset=['game_date', 'pitcher'], inplace=True)
                        all_chunks.append(chunk)
            else: logging.warning("Skipping loading: no rows estimated.")
    except Exception as e:
        logging.error(f"Error during data loading: {e}", exc_info=True)
        return pd.DataFrame()

    if not all_chunks: return pd.DataFrame()
    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Successfully loaded {len(df)} rows.")
    return df


def calculate_pitcher_target_and_stats(df):
    """Aggregates pitch data to get actual SO (target), Outs, Walks, and Pitch Mix per pitcher-game."""
    logging.info("Calculating target variable (SO) and base stats for pitchers...")
    if df.empty: return pd.DataFrame()

    required = ['game_date', 'game_pk', 'pitcher', 'events', 'pitch_type']
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        logging.error(f"Missing required columns for pitcher aggregation: {missing}")
        return pd.DataFrame()

    # Convert IDs
    for id_col in ['pitcher', 'game_pk']:
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
    df.dropna(subset=['pitcher', 'game_pk', 'game_date'], inplace=True)
    df['game_pk'] = df['game_pk'].astype(int)

    # --- Event Flags ---
    df['events'] = df['events'].astype(str)
    df['is_strikeout'] = (df['events'] == 'strikeout').astype(int)
    if 'description' in df.columns:
        desc = df['description'].astype(str).str.lower()
        df.loc[desc.str.contains('strikes out', na=False), 'is_strikeout'] = 1
    df['is_walk'] = (df['events'] == 'walk').astype(int)

    if 'is_strikeout' not in df.columns:
         logging.error("FATAL: 'is_strikeout' column failed to be created or was lost.")
         return pd.DataFrame()
    logging.debug(f"Column 'is_strikeout' created. Exists: {'is_strikeout' in df.columns}")

    # --- Outs Recorded ---
    df['outs_made'] = 0
    event_out_map = {'strikeout': 1, 'field_out': 1, 'sac_fly': 1, 'sac_bunt': 1, 'force_out': 1,'grounded_into_double_play': 2, 'double_play': 2, 'triple_play': 3,'fielders_choice_out': 1, 'caught_stealing_2b': 1, 'caught_stealing_3b': 1, 'caught_stealing_home': 1,'pickoff_1b': 1, 'pickoff_2b': 1, 'pickoff_3b': 1, 'pickoff_caught_stealing_2b': 1,'pickoff_caught_stealing_3b': 1, 'pickoff_caught_stealing_home': 1, 'other_out': 1,'sac_fly_double_play': 2, 'sac_bunt_double_play': 2, 'strikeout_double_play': 2}
    df['outs_made'] = df['events'].map(event_out_map).fillna(0)
    if 'description' in df.columns:
        desc = df['description'].astype(str).str.lower()
        df.loc[desc.str.contains('double play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 2)
        df.loc[desc.str.contains('triple play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 3)
    df['outs_made'] = df['outs_made'].astype(int)

    # --- Pitch Mix ---
    df['pitch_category'] = df['pitch_type'].map(PITCH_CLASSIFICATION).fillna('OTHER')
    pitch_dummies = pd.get_dummies(df['pitch_category'], prefix='pitch', dtype=int)
    for col in pitch_dummies.columns: df[col] = pitch_dummies[col]
    pitch_dummy_cols_to_agg = list(pitch_dummies.columns)

    # --- Aggregation ---
    group_cols = ['game_date', 'game_pk', 'pitcher'] # Grouping keys
    agg_ops = {
        # Use a column that exists for 'size' aggregation
        'game_pk': 'size', # <<-- Aggregate using game_pk directly
        'is_strikeout': 'sum',
        'is_walk': 'sum',
        'outs_made': 'sum',
        **{col: 'sum' for col in pitch_dummy_cols_to_agg}
    }
    # Filter based on source columns existing in df
    valid_agg_ops = {}
    for k, v in agg_ops.items():
        # If v is a string ('sum'), the key 'k' is the column to aggregate
        # If v is a tuple ('col', 'size'), the first element v[0] is the column to aggregate on
        source_col = k if isinstance(v, str) else v[0]
        # Check if the source column exists OR if the key is a pitch dummy column
        if (source_col in df.columns) or (k.startswith('pitch_') and k in df.columns):
            valid_agg_ops[k] = v

    # Re-check if 'is_strikeout' is a valid aggregation target *after* filtering
    if 'is_strikeout' not in valid_agg_ops:
        logging.error("Cannot calculate target variable 'SO'. Aggregation setup failed for 'is_strikeout'.")
        logging.error(f"Columns present in df before agg: {df.columns.tolist()}")
        logging.error(f"Final Agg Ops Dict: {valid_agg_ops}")
        return pd.DataFrame()

    # Perform aggregation
    daily_stats = df.groupby(group_cols, as_index=False, observed=True).agg(valid_agg_ops)

    # Rename columns after aggregation
    rename_dict = {
        'game_pk': 'Total_Pitches', # <<-- Rename the aggregated game_pk count
        'is_strikeout': 'SO',
        'is_walk': 'BB',
        'outs_made': 'Outs'
    }
    daily_stats.rename(columns=rename_dict, inplace=True)

    if 'pitcher' not in daily_stats.columns:
         logging.error("Pitcher ID column was lost during aggregation!")
         return pd.DataFrame()

    # Calculate Pitch Mix Percentages
    pitch_cat_cols = [col for col in daily_stats.columns if col.startswith('pitch_')]
    for col in pitch_cat_cols:
        pct_col_name = col.replace('pitch_', '') + '_pct'
        total_pitches_numeric = safe_get_numeric(daily_stats['Total_Pitches'], default=1)
        daily_stats[pct_col_name] = np.where(
            total_pitches_numeric > 0,
            safe_get_numeric(daily_stats[col]) / total_pitches_numeric,
            0.0
        )

    # Ensure integer types for counts
    int_cols = ['SO', 'BB', 'Outs', 'Total_Pitches'] + pitch_cat_cols
    for col in int_cols:
        if col in daily_stats.columns:
             daily_stats[col] = daily_stats[col].fillna(0).astype(int)

    # Ensure pitcher ID is int before returning
    daily_stats['pitcher'] = daily_stats['pitcher'].astype(int)

    logging.info(f"Calculated target and base stats for {len(daily_stats)} pitcher-game instances.")
    return daily_stats


def create_pitcher_features(df):
    """Creates rolling and lagged features for predicting pitcher SO."""
    logging.info("Creating features for Pitcher SO model...")
    if df.empty: return pd.DataFrame(), []
    # ** Check for pitcher column here **
    required_feature_cols = ['pitcher', 'game_date', 'game_pk', 'SO', 'BB', 'Outs', 'Total_Pitches'] + [col for col in df.columns if col.endswith('_pct')]
    if not all(c in df.columns for c in required_feature_cols):
        missing = [c for c in required_feature_cols if c not in df.columns]
        logging.error(f"Missing required columns for feature engineering: {missing}")
        return pd.DataFrame(), []

    # Ensure correct dtypes before sorting
    df['pitcher'] = pd.to_numeric(df['pitcher'], errors='coerce').astype('Int64')
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce').astype('Int64')
    df.dropna(subset=['pitcher', 'game_date', 'game_pk'], inplace=True)
    df['pitcher'] = df['pitcher'].astype(int)
    df['game_pk'] = df['game_pk'].astype(int)

    df = df.sort_values(by=['pitcher', 'game_date', 'game_pk']).copy()

    # --- Calculate Base Rates ---
    df['K_per_9'] = np.where(df['Outs'] > 0, df['SO'] * 27 / df['Outs'], 0.0)
    df['BB_per_9'] = np.where(df['Outs'] > 0, df['BB'] * 27 / df['Outs'], 0.0)

    # --- Calculate Rolling Features ---
    features = []
    grouped = df.groupby('pitcher') # Group by the correct pitcher ID column

    # --- Rolling K/9 and BB/9 ---
    for G in ROLLING_GAMES_BASE:
        min_g = max(1, G // 3)
        df[f'roll_{G}g_SO_sum'] = grouped['SO'].transform(lambda x: x.rolling(G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_BB_sum'] = grouped['BB'].transform(lambda x: x.rolling(G, min_periods=min_g).sum().shift(1))
        df[f'roll_{G}g_Outs_sum'] = grouped['Outs'].transform(lambda x: x.rolling(G, min_periods=min_g).sum().shift(1))

        k9_col, bb9_col = f'roll_{G}g_K_per_9', f'roll_{G}g_BB_per_9'
        roll_outs = safe_get_numeric(df[f'roll_{G}g_Outs_sum'])
        df[k9_col] = np.where(roll_outs > 0, safe_get_numeric(df[f'roll_{G}g_SO_sum']) * 27 / roll_outs, 0.0)
        df[bb9_col] = np.where(roll_outs > 0, safe_get_numeric(df[f'roll_{G}g_BB_sum']) * 27 / roll_outs, 0.0)
        features.extend([k9_col, bb9_col])

    # --- Rolling Pitch Mix % ---
    pitch_pct_cols = [col for col in df.columns if col.endswith('_pct')]
    for G in ROLLING_GAMES_PITCHMIX:
        min_g = max(1, G // 3)
        for col in pitch_pct_cols:
            roll_col_name = f'roll_{G}g_{col}'
            df[roll_col_name] = grouped[col].transform(lambda x: x.rolling(G, min_periods=min_g).mean().shift(1))
            features.append(roll_col_name)

    # --- Off-speed Uptick Ratio ---
    short_offspeed_col = f'roll_{ROLLING_GAMES_PITCHMIX[0]}g_OFFSPEED_pct'
    long_offspeed_col = f'roll_{ROLLING_GAMES_PITCHMIX[-1]}g_OFFSPEED_pct'
    uptick_col = 'Offspeed_Uptick_Ratio'
    if short_offspeed_col in df.columns and long_offspeed_col in df.columns:
        long_pct = safe_get_numeric(df[long_offspeed_col])
        df[uptick_col] = np.where(long_pct > 0.01, safe_get_numeric(df[short_offspeed_col]) / long_pct, 1.0)
        features.append(uptick_col)
    else: logging.warning("Could not calculate Offspeed_Uptick_Ratio.")

    # --- New Pitch Flag Feature ---
    new_pitch_col = 'new_pitch_flag'
    df[new_pitch_col] = 0
    baseline_window = ROLLING_GAMES_PITCHMIX[-1]
    for pct_col in pitch_pct_cols:
         baseline_roll_col = f'roll_{baseline_window}g_{pct_col}'
         if baseline_roll_col in df.columns:
              is_new = (safe_get_numeric(df[baseline_roll_col]) <= 0.01) & (safe_get_numeric(df[pct_col]) > 0.02)
              df.loc[is_new, new_pitch_col] = 1
    features.append(new_pitch_col)

    # --- Lagged Features ---
    df['lag_1g_SO'] = grouped['SO'].shift(1)
    df['lag_1g_Outs'] = grouped['Outs'].shift(1)
    features.extend(['lag_1g_SO', 'lag_1g_Outs'])

    # --- TODO: Add Park Factor, Opponent Features ---
    df['park_factor_SO'] = 1.0 # Placeholder
    df['opponent_K_pct'] = 0.22 # Placeholder league avg K%
    features.extend(['park_factor_SO', 'opponent_K_pct'])

    # Fill NaNs created by rolling/lagging
    df[features] = df[features].fillna(0)

    # Clean up intermediate columns
    cols_to_drop = [c for c in df.columns if '_sum' in c or c in ['K_per_9', 'BB_per_9']]
    # cols_to_drop.extend(pitch_pct_cols) # Keep pitch percentages for inspection
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    logging.info(f"Created {len(features)} features: {features}")
    return df, features


def train_and_evaluate(df, features, target_col='SO'):
    """Splits data, trains XGBoost model, evaluates, and saves."""
    logging.info("Splitting data, training model, and evaluating...")
    if df.empty or not features: return

    # Time-based split
    df['game_date'] = pd.to_datetime(df['game_date'])
    train_df = df[df['game_date'] < TEST_SET_START_DATE].copy()
    test_df = df[df['game_date'] >= TEST_SET_START_DATE].copy()

    if train_df.empty or test_df.empty: return

    missing_features_train = [f for f in features if f not in train_df.columns]
    missing_features_test = [f for f in features if f not in test_df.columns]
    if missing_features_train or missing_features_test:
         logging.error(f"Missing expected features! Train missing: {missing_features_train}, Test missing: {missing_features_test}")
         return

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # --- Model Training ---
    logging.info("Training XGBoost Regressor model for SO...")
    model = XGBRegressor(
        objective='count:poisson', n_estimators=200, learning_rate=0.07,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, early_stopping_rounds=15
    )
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    except Exception as e: logging.error(f"Error during model training: {e}", exc_info=True); return

    logging.info("Model training complete.")
    logging.info(f"Best Iteration: {model.best_iteration}")

    # --- Evaluation ---
    predictions = model.predict(X_test)
    predictions_non_negative = np.maximum(0, predictions)
    predictions_rounded = np.round(predictions_non_negative).astype(int)
    mae = mean_absolute_error(y_test, predictions_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_rounded))
    r2 = r2_score(y_test, predictions)
    logging.info("--- Pitcher SO Model Evaluation ---")
    logging.info(f"Target: {target_col}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    logging.info("---------------------------------")

    # --- Feature Importance ---
    try:
        feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        logging.info("Top 10 Feature Importances:")
        logging.info(feature_importances.head(10).to_string(index=False))
    except Exception as e: logging.warning(f"Could not get feature importances: {e}")

    # --- Save Model and Features ---
    try:
        with open(MODEL_OUTPUT_PATH, 'wb') as f: pickle.dump(model, f)
        logging.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        with open(FEATURES_OUTPUT_PATH, 'wb') as f: pickle.dump(features, f)
        logging.info(f"Feature list saved to {FEATURES_OUTPUT_PATH}")
    except Exception as e: logging.error(f"Error saving model/features: {e}", exc_info=True)


# === Main Execution ===
if __name__ == "__main__":
    logging.info("--- Starting Pitcher SO Model Training ---")
    start_time = time.time()
    raw_data = load_data_for_pitcher_so(DATA_START_DATE, DATA_END_DATE)
    if not raw_data.empty:
        daily_stats_df = calculate_pitcher_target_and_stats(raw_data)
        del raw_data; gc.collect()
        if not daily_stats_df.empty:
            features_df, feature_names = create_pitcher_features(daily_stats_df)
            del daily_stats_df; gc.collect()
            if not features_df.empty and feature_names:
                train_and_evaluate(features_df, feature_names, target_col='SO')
            else: logging.error("Feature creation failed.")
        else: logging.error("Target variable calculation failed.")
    else: logging.error("Data loading failed.")
    end_time = time.time()
    logging.info(f"--- Script Finished ---")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
