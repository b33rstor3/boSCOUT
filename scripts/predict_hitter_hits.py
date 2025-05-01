# predict_hitter_hits.py
# Loads the trained model and predicts hitter hits for a given date.

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
import requests # For schedule fetching
# No need to import XGBRegressor or metrics here, just loading the model

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None # default='warn'

# === Configuration & Constants ===
DB_PATH = "mlb_data2.db"
TABLE_NAME = "pitch_data"
LOG_FILE = "predict_hitter_hits.log"
MODEL_PATH = "hitter_hits_model.pkl"
FEATURES_PATH = "hitter_hits_features.pkl"
OUTPUT_CSV = "hitter_hits_predictions.csv"

# --- Prediction Target Date ---
# Default to tomorrow, but can be changed
# PREDICTION_DATE_STR = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
PREDICTION_DATE_STR = "2023-10-01" # Example: Predict for the last day of sample data

CHUNKSIZE = 20000

# Feature Engineering Parameters (should match training script)
ROLLING_GAMES = [15, 30]
HISTORICAL_DAYS_NEEDED = 60

# MLB API Constants
MLB_SCHEDULE_API_URL = "https://statsapi.mlb.com/api/v1/schedule"
TEAM_ABBR_MAP = { # Map API abbreviations if needed
    'CWS': 'CWS', 'CHW': 'CWS', 'CHC': 'CHC', 'NYM': 'NYM', 'NYY': 'NYY',
    'LAA': 'LAA', 'ANA': 'LAA', 'SD': 'SD', 'SDP': 'SD', 'SF': 'SF', 'SFG': 'SF',
    'TB': 'TB', 'TBR': 'TB', 'KC': 'KC', 'KCR': 'KC', 'WSH': 'WSH', 'WSN': 'WSH',
}


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

# === Feature Engineering Function (Copied from Training Script) ===
def create_hitter_features(df):
    """Creates rolling and lagged features for predicting hits."""
    logging.info("Creating features for Hitter Hits model...")
    if df.empty: return pd.DataFrame(), []
    required_feature_cols = ['batter', 'game_date', 'game_pk', 'AB', 'PA', 'BB', 'SO', 'H']
    if not all(c in df.columns for c in required_feature_cols):
        missing_cols = [c for c in required_feature_cols if c not in df.columns]
        logging.error(f"Missing required columns for feature engineering: {missing_cols}")
        return pd.DataFrame(), []

    # Ensure correct dtypes before sorting
    df['batter'] = pd.to_numeric(df['batter'], errors='coerce').astype('Int64')
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df['game_pk'] = pd.to_numeric(df['game_pk'], errors='coerce').astype('Int64')
    df.dropna(subset=['batter', 'game_date', 'game_pk'], inplace=True)
    df['batter'] = df['batter'].astype(int)
    df['game_pk'] = df['game_pk'].astype(int)


    df = df.sort_values(by=['batter', 'game_date', 'game_pk']).copy()

    # Calculate base rates needed for rolling averages
    # Use np.where to avoid division by zero safely
    df['AVG'] = np.where(df['AB'] > 0, df['H'] / df['AB'], 0.0)
    df['K_pct'] = np.where(df['PA'] > 0, df['SO'] / df['PA'], 0.0)
    df['BB_pct'] = np.where(df['PA'] > 0, df['BB'] / df['PA'], 0.0)


    # --- Calculate Rolling Features ---
    features = []
    # Group by batter ID after sorting
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
    # Ensure these match exactly what was used in training
    df['park_factor_H'] = 1.0 # Neutral park factor placeholder
    df['vs_LHP'] = 0 # Indicator placeholder
    df['vs_RHP'] = 0 # Indicator placeholder
    features.extend(['park_factor_H', 'vs_LHP', 'vs_RHP'])

    # Fill NaNs created by rolling/lagging (use 0 for simplicity)
    # Important: Only fill NaNs in the feature columns themselves
    df[features] = df[features].fillna(0)

    logging.info(f"Created {len(features)} features.")
    # Return the dataframe with features and the list of feature names
    return df, features

# === Core Logic Functions ===

def load_required_data(end_date_dt):
    """Loads historical data needed for feature calculation up to the day BEFORE prediction."""
    start_date_dt = end_date_dt - timedelta(days=HISTORICAL_DAYS_NEEDED)
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    # Load data UP TO THE DAY BEFORE the prediction date
    end_date_str = (end_date_dt - timedelta(days=1)).strftime('%Y-%m-%d')

    logging.info(f"Loading historical pitch data from {start_date_str} to {end_date_str} for feature calculation.")
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found: {DB_PATH}")
        return pd.DataFrame()

    # Columns needed for base stats and features
    required_cols = [
        'game_date', 'game_pk', 'batter', 'events', 'description',
        'woba_denom', 'bb_type',
        # Add columns needed for features (e.g., pitcher, park info if used)
        'home_team', 'away_team'
    ]

    # --- Simplified loading - adapt from train script if needed ---
    select_cols_str = ', '.join([f'"{c}"' for c in required_cols if c != 'is_ab']) # Select needed cols
    query = f"SELECT {select_cols_str} FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
    params = (start_date_str, end_date_str)

    all_chunks = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Simplified: No row count for prediction loading
            chunk_iterator = pd.read_sql_query(query, conn, params=params, chunksize=CHUNKSIZE)
            for chunk in tqdm(chunk_iterator, desc="Loading Hist Data"):
                if not chunk.empty:
                    chunk['game_date'] = pd.to_datetime(chunk['game_date'], errors='coerce')
                    chunk.dropna(subset=['game_date'], inplace=True)
                    all_chunks.append(chunk)
    except Exception as e:
        logging.error(f"Error loading historical data: {e}", exc_info=True)
        return pd.DataFrame()

    if not all_chunks: return pd.DataFrame()
    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Successfully loaded {len(df)} historical rows.")
    return df


def calculate_base_stats_for_features(df):
    """Aggregates historical data to get base stats needed for feature calculation."""
    # This is similar to calculate_target_and_base_stats, but we don't need the target 'H' here
    logging.info("Calculating base stats from historical data...")
    if df.empty: return pd.DataFrame()

    required = ['game_date', 'game_pk', 'batter', 'events', 'woba_denom', 'bb_type']
    if not all(c in df.columns for c in required):
        missing = [c for c in required if c not in df.columns]
        logging.error(f"Missing required columns for base stat calculation: {missing}")
        return pd.DataFrame()


    for id_col in ['batter', 'game_pk']:
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')
    df.dropna(subset=['batter', 'game_pk', 'game_date'], inplace=True)
    df['batter'] = df['batter'].astype(int)
    df['game_pk'] = df['game_pk'].astype(int)

    # PA calculation
    if 'woba_denom' in df.columns and df['woba_denom'].notna().any():
        df['is_pa_outcome'] = (safe_get_numeric(df['woba_denom'], 0) == 1).astype(int)
    else:
        pa_ending_events = ['strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run','field_out', 'sac_fly', 'sac_bunt', 'force_out', 'grounded_into_double_play','fielders_choice', 'field_error', 'double_play', 'triple_play', 'batter_interference','catcher_interf']
        df['is_pa_outcome'] = df['events'].astype(str).isin(pa_ending_events).astype(int)

    # Event Flags
    event_flags = {'single': 'single', 'double': 'double', 'triple': 'triple', 'home_run': 'home_run',
                   'walk': 'walk', 'hit_by_pitch': 'hit_by_pitch', 'strikeout': 'strikeout',
                   'sac_fly': 'sac_fly', 'sac_bunt': 'sac_bunt'}
    events_str = df['events'].astype(str)
    for event, flag_col in event_flags.items():
        if flag_col not in df.columns: df[flag_col] = 0
        df[flag_col] = (events_str == event).astype(int)
    if 'description' in df.columns:
        desc = df['description'].astype(str).str.lower()
        if 'strikeout' not in df.columns: df['strikeout'] = 0
        df.loc[desc.str.contains('strikes out', na=False), 'strikeout'] = 1

    # AB calculation
    df['is_walk'] = df['walk']
    df['is_hbp'] = df['hit_by_pitch']
    df['is_sh'] = df['sac_bunt']
    df['is_sf'] = df['sac_fly']
    df['is_ci'] = (events_str == 'catcher_interf').astype(int)
    df['non_ab_event'] = df['is_walk'] + df['is_hbp'] + df['is_sh'] + df['is_sf'] + df['is_ci']
    df['is_ab'] = ((df['is_pa_outcome'] == 1) & (df['non_ab_event'] == 0)).astype(int)
    df['H'] = df['single'] + df['double'] + df['triple'] + df['home_run'] # Hits needed for rolling AVG

    # Aggregation
    group_cols = ['game_date', 'game_pk', 'batter']
    agg_ops = {
        'H': 'sum', 'is_pa_outcome': 'sum', 'is_ab': 'sum',
        'is_walk': 'sum', 'strikeout': 'sum',
    }
    valid_agg_ops = {k: v for k, v in agg_ops.items() if k in df.columns}
    daily_stats = df.groupby(group_cols, as_index=False, observed=True).agg(valid_agg_ops)
    daily_stats.rename(columns={'is_pa_outcome': 'PA', 'is_ab': 'AB', 'is_walk': 'BB', 'strikeout': 'SO'}, inplace=True)

    # Ensure integer types
    int_cols = ['H', 'PA', 'AB', 'BB', 'SO']
    for col in int_cols:
        if col in daily_stats.columns:
             daily_stats[col] = daily_stats[col].astype(int)

    logging.info(f"Calculated base stats for {len(daily_stats)} historical batter-game instances.")
    return daily_stats


def get_scheduled_batters(prediction_date_dt):
    """
    Fetches schedule and attempts to identify likely batters.
    NOTE: This is a placeholder/approximation as full lineups aren't usually available early.
    """
    logging.info(f"Fetching schedule for {prediction_date_dt.strftime('%Y-%m-%d')} to identify potential batters...")
    # --- Reusing schedule fetching logic ---
    formatted_date = prediction_date_dt.strftime('%Y-%m-%d')
    params = {'sportId': 1, 'startDate': formatted_date, 'endDate': formatted_date}
    scheduled_teams = set()
    try:
        response = requests.get(MLB_SCHEDULE_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'dates' in data and data['dates']:
            games = data['dates'][0].get('games', [])
            for game in games:
                home_team_id = game.get('teams', {}).get('home', {}).get('team', {}).get('id')
                away_team_id = game.get('teams', {}).get('away', {}).get('team', {}).get('id')
                if home_team_id: scheduled_teams.add(home_team_id)
                if away_team_id: scheduled_teams.add(away_team_id)
    except Exception as e:
        logging.error(f"Failed to fetch schedule: {e}")
        return set() # Return empty set on error

    logging.info(f"Found {len(scheduled_teams)} teams scheduled to play.")

    # --- Placeholder: Get recent batters ---
    if not os.path.exists(DB_PATH):
        return set()

    recent_batters = set()
    lookback_days = 7 # How far back to look for active players
    start_lookback = (prediction_date_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    end_lookback = (prediction_date_dt - timedelta(days=1)).strftime('%Y-%m-%d')

    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = f"""
                SELECT DISTINCT batter
                FROM {TABLE_NAME}
                WHERE game_date BETWEEN ? AND ?
                  AND batter IS NOT NULL
            """
            params = (start_lookback, end_lookback)
            recent_batters_df = pd.read_sql_query(query, conn, params=params)
            recent_batters = set(pd.to_numeric(recent_batters_df['batter'], errors='coerce').dropna().astype(int))
            logging.info(f"Identified {len(recent_batters)} batters active in the last {lookback_days} days.")
            # Filter by scheduled teams (approximation - needs team ID mapping)
            # For now, returning all recent batters as potential players
            return recent_batters
    except Exception as e:
        logging.error(f"Error fetching recent batters: {e}")
        return set()


# === Main Execution ===
if __name__ == "__main__":
    logging.info("--- Starting Hitter Hits Prediction ---")
    start_time = time.time()

    try:
        prediction_date = datetime.strptime(PREDICTION_DATE_STR, '%Y-%m-%d').date()
    except ValueError:
        logging.error(f"Invalid PREDICTION_DATE_STR format: {PREDICTION_DATE_STR}. Use YYYY-MM-DD.")
        exit()

    # 1. Load Model and Features
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f: feature_names = pickle.load(f)
        logging.info(f"Loaded model from {MODEL_PATH} and features from {FEATURES_PATH}")
    except FileNotFoundError:
        logging.error(f"Model ({MODEL_PATH}) or features ({FEATURES_PATH}) file not found. Train the model first.")
        exit()
    except Exception as e:
        logging.error(f"Error loading model/features: {e}", exc_info=True)
        exit()

    # 2. Load Historical Data
    hist_data_df = load_required_data(prediction_date)

    if not hist_data_df.empty:
        # 3. Calculate Base Stats from History
        daily_stats_hist = calculate_base_stats_for_features(hist_data_df)
        del hist_data_df; gc.collect()

        if not daily_stats_hist.empty:
            # 4. Create Features for Prediction Date
            # Pass the historical daily stats to the feature creation function
            features_df, used_features = create_hitter_features(daily_stats_hist) # Use the function defined above
            del daily_stats_hist; gc.collect()

            # Filter features_df to include only the latest available features for each player *before* the prediction date
            features_df['game_date'] = pd.to_datetime(features_df['game_date'])
            # Ensure the date conversion worked
            features_df.dropna(subset=['game_date'], inplace=True)

            # Get features for the day *before* the prediction date
            latest_feature_date = prediction_date - timedelta(days=1)
            latest_features = features_df[features_df['game_date'] <= pd.to_datetime(latest_feature_date)].copy()
            # Get the single most recent row of features per player
            latest_features = latest_features.sort_values('game_date').groupby('batter').tail(1)


            if not latest_features.empty and feature_names:
                # 5. Identify Players to Predict For (Using latest features as proxy)
                players_to_predict = latest_features['batter'].unique()
                logging.info(f"Predicting for {len(players_to_predict)} batters with recent data.")

                # Ensure all necessary features are present in the prediction set
                missing_model_features = [f for f in feature_names if f not in latest_features.columns]
                if missing_model_features:
                    logging.error(f"Features mismatch! Model requires {missing_model_features} which are not in the generated features.")
                    for f in missing_model_features: latest_features[f] = 0 # Fill missing with 0

                # Select only the required features in the correct order
                X_predict = latest_features[feature_names].fillna(0)

                # 6. Make Predictions
                try:
                    predictions_raw = model.predict(X_predict)
                    predictions = np.maximum(0, predictions_raw) # Ensure non-negative
                    logging.info("Predictions generated successfully.")
                except Exception as e:
                    logging.error(f"Error during prediction: {e}", exc_info=True)
                    predictions = None

                if predictions is not None:
                    # 7. Format and Save Output
                    output_df = pd.DataFrame({
                        'player_id': latest_features['batter'],
                        'prediction_date': PREDICTION_DATE_STR,
                        'predicted_hits_raw': predictions,
                        'predicted_hits_rounded': np.round(predictions).astype(int)
                    })

                    output_df = output_df.sort_values(by='predicted_hits_raw', ascending=False)

                    try:
                        output_df.to_csv(OUTPUT_CSV, index=False)
                        logging.info(f"Predictions saved to {OUTPUT_CSV}")
                        print("\n--- Top Predicted Hitters (Hits) ---")
                        print(output_df.head(20).to_string(index=False))
                        print("------------------------------------\n")
                    except Exception as e:
                        logging.error(f"Error saving predictions to CSV: {e}")

            else:
                logging.error("Feature creation for prediction failed or resulted in empty data.")
        else:
            logging.error("Calculation of base stats from historical data failed.")
    else:
        logging.error("Loading historical data for feature calculation failed.")

    end_time = time.time()
    logging.info(f"--- Prediction Script Finished ---")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
