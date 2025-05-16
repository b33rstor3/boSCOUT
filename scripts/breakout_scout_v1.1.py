# breakout_scout_v1.1.py
# Module to identify historical breakout performances from MLB data,
# selecting top N per day based on game count.

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import time
import os
import traceback
import warnings
import pickle # To load feature lists if needed for context
from tqdm import tqdm
import gc # Garbage Collection

# Attempt to import meteostat for historical weather
try:
    from meteostat import Point, Daily
except ImportError:
    logging.warning("meteostat library not found. Historical weather features will use defaults or require manual addition. Install with: pip install meteostat")
    Point, Daily = None, None

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress SettingWithCopyWarning, though care should be taken
pd.options.mode.chained_assignment = None # default='warn'

# === Configuration & Constants ===
DB_PATH = "mlb_data2.db" # Path to the SQLite database
TABLE_NAME = "pitch_data" # Table containing pitch-by-pitch data
OUTPUT_DB_PATH = "breakouts.db" # Database to store identified breakouts
OUTPUT_TABLE_NAME = "breakout_performances_topN" # Changed table name for clarity
LOG_FILE = "breakout_scout.log"

# Date range for historical analysis (adjust as needed)
ANALYSIS_START_DATE = "2023-03-30"
ANALYSIS_END_DATE = "2023-10-01"

CHUNKSIZE = 10000 # Process data in chunks to manage memory

# Breakout Definition Parameters
ROLLING_WINDOW_DAYS = 30 # Look back period for rolling stats/percentiles
PERFORMANCE_PERCENTILE_THRESHOLD = 95 # Performance must be > this percentile of player's recent history
ROLLING_AVG_METRIC = 'daily_score' # Metric to use for rolling average comparison
ROLLING_AVG_THRESHOLD_TYPE = 'player_median' # Options: 'player_median', 'league_median', 'fixed_value'

# Superstar Filter (Example IDs - Replace with actual relevant IDs)
SUPERSTAR_PLAYER_IDS = {
    592450, # Aaron Judge
    660271, # Shohei Ohtani
    545361, # Mookie Betts
    605141, # Bryce Harper
    518692, # Gerrit Cole
    # Add more IDs as needed
}

# Daily Selection Parameters
GAMES_THRESHOLD = 3
TOP_N_HIGH_GAMES = 25
TOP_N_LOW_GAMES = 10

# League averages (can be loaded dynamically or use placeholders)
LEAGUE_AVERAGES = {
    'release_speed': 93.0, 'spin_rate': 2400.0, 'launch_speed': 88.0,
    'launch_angle': 12.0, 'estimated_ba_using_speedangle': 0.245,
    'estimated_woba_using_speedangle': 0.315, 'pfx_x': 0.0, 'pfx_z': 0.5,
    'temp': 70.0, 'humidity': 50.0, 'wind_speed': 5.0, 'wind_dir': 0.0,
    'pressure': 1013.25, 'league_woba': 0.315, 'wOBA_scale': 1.2,
    'FIP_constant': 3.10, 'lgFIP': 3.10, 'runs_per_win': 9.0,
    'outs_recorded': 0, 'home_run': 0, 'walk': 0, 'hit_by_pitch': 0,
    'strikeout': 0, 'PA': 0, 'woba_value': 0, 'daily_wpa': 0,
    'single': 0, 'double': 0, 'triple': 0, 'is_barrel': 0,
    'release_spin_rate': 2400.0
}

# Ballpark coordinates (copied from MLb_v3.4.2.py)
BALLPARK_COORDS = {
    'ARI': {'lat': 33.4456, 'lon': -112.0667}, 'ATL': {'lat': 33.8908, 'lon': -84.4678},
    'BAL': {'lat': 39.2838, 'lon': -76.6217}, 'BOS': {'lat': 42.3467, 'lon': -71.0972},
    'CHC': {'lat': 41.9484, 'lon': -87.6553}, 'CWS': {'lat': 41.8300, 'lon': -87.6338},
    'CIN': {'lat': 39.0974, 'lon': -84.5070}, 'CLE': {'lat': 41.4962, 'lon': -81.6852},
    'COL': {'lat': 39.7562, 'lon': -104.9942}, 'DET': {'lat': 42.3390, 'lon': -83.0485},
    'HOU': {'lat': 29.7573, 'lon': -95.3555}, 'KC': {'lat': 39.0517, 'lon': -94.4803},
    'LAA': {'lat': 33.8003, 'lon': -117.8827}, 'LAD': {'lat': 34.0739, 'lon': -118.2400},
    'MIA': {'lat': 25.7781, 'lon': -80.2196}, 'MIL': {'lat': 43.0280, 'lon': -87.9712},
    'MIN': {'lat': 44.9817, 'lon': -93.2777}, 'NYM': {'lat': 40.7571, 'lon': -73.8458},
    'NYY': {'lat': 40.8296, 'lon': -73.9262}, 'OAK': {'lat': 37.7510, 'lon': -122.2009},
    'PHI': {'lat': 39.9061, 'lon': -75.1665}, 'PIT': {'lat': 40.4469, 'lon': -80.0057},
    'SD': {'lat': 32.7076, 'lon': -117.1570}, 'SEA': {'lat': 47.5914, 'lon': -122.3325},
    'SF': {'lat': 37.7786, 'lon': -122.3893}, 'STL': {'lat': 38.6226, 'lon': -90.1928},
    'TB': {'lat': 27.7682, 'lon': -82.6534}, 'TEX': {'lat': 32.7513, 'lon': -97.0828},
    'TOR': {'lat': 43.6414, 'lon': -79.3894}, 'WSH': {'lat': 38.8730, 'lon': -77.0074},
    'DEFAULT': {'lat': 39.8283, 'lon': -98.5795}
}
TEAM_ABBR_MAP = { # Map API abbreviations to Ballpark Coords keys if needed
    'CWS': 'CWS', 'CHW': 'CWS', 'CHC': 'CHC', 'NYM': 'NYM', 'NYY': 'NYY',
    'LAA': 'LAA', 'ANA': 'LAA', 'SD': 'SD', 'SDP': 'SD', 'SF': 'SF', 'SFG': 'SF',
    'TB': 'TB', 'TBR': 'TB', 'KC': 'KC', 'KCR': 'KC', 'WSH': 'WSH', 'WSN': 'WSH',
}

# === Logging Setup ===
# Remove existing handlers to avoid duplicate logs if script is re-run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE, mode='w') # Overwrite log file each run
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# === Helper Functions ===
def safe_get_numeric(series, default=0.0):
    """Convert series to numeric, coercing errors and filling NaNs with a default."""
    if series is None:
        return pd.Series(dtype=float) # Return empty Series if input is None
    if series.empty:
        return series # Return empty series if input is empty
    return pd.to_numeric(series, errors='coerce').fillna(default)

# === Core Logic Functions ===

def load_historical_data(start_date_str, end_date_str):
    """
    Loads pitch data from the SQLite database for the specified date range using chunking.
    Selects only necessary columns for aggregation and scoring.
    """
    logging.info(f"Loading pitch data from {start_date_str} to {end_date_str} from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found: {DB_PATH}")
        return pd.DataFrame()

    # Define columns needed for aggregation, scoring, and game counting
    required_cols = [
        'game_date', 'batter', 'pitcher', 'events', 'description', 'inning_topbot',
        'home_team', 'away_team', 'game_pk', 'at_bat_number', 'pitch_type',
        'delta_home_win_exp', 'woba_value', 'woba_denom',
        'launch_speed', 'launch_angle', 'release_speed', 'release_spin_rate', # Use consistent name
        'pfx_x', 'pfx_z', 'is_barrel' # Include pre-calculated barrel if exists
    ]
    # Alias 'spin_rate' if 'release_spin_rate' isn't present
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
    actual_cols_needed = []
    has_release_spin_rate = 'release_spin_rate' in available_cols
    has_spin_rate = 'spin_rate' in available_cols

    for col in required_cols:
        if col in available_cols:
            select_cols_sql.append(f'"{col}"')
            actual_cols_needed.append(col)
        elif col == 'release_spin_rate' and has_spin_rate:
            select_cols_sql.append(f'"spin_rate" AS "release_spin_rate"')
            actual_cols_needed.append('release_spin_rate')
            logging.debug("Mapping DB column 'spin_rate' to 'release_spin_rate'.")
        elif col == 'is_barrel':
             logging.debug("'is_barrel' column not found in DB, will calculate if possible.")
        else:
             logging.warning(f"Required column '{col}' not found in database table '{TABLE_NAME}'. It will be skipped or defaulted.")

    # Ensure game_pk is selected as it's needed for game counting
    if 'game_pk' not in actual_cols_needed and 'game_pk' in available_cols:
         select_cols_sql.append('"game_pk"')
         actual_cols_needed.append('game_pk')
    elif 'game_pk' not in available_cols:
         logging.error("Essential column 'game_pk' not found in database. Cannot count games per day.")
         pass # Allow script to continue, error will occur later

    if not all(c in actual_cols_needed + ['spin_rate'] for c in ['game_date', 'batter', 'pitcher', 'events']):
        logging.error("Missing one or more essential columns (game_date, batter, pitcher, events). Cannot proceed.")
        return pd.DataFrame()

    select_cols_str = ', '.join(list(set(select_cols_sql))) # Use set to remove duplicates if any column added twice
    query = f"SELECT {select_cols_str} FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
    params = (start_date_str, end_date_str)
    logging.debug(f"Executing query: {query} with params {params}")


    all_chunks = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Estimate total rows
            total_rows = 0
            n_chunks = None
            try:
                count_cursor = conn.cursor()
                count_query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
                count_cursor.execute(count_query, params)
                result = count_cursor.fetchone()
                total_rows = result[0] if result else 0
                count_cursor.close()
                if total_rows > 0:
                    n_chunks = max(1, (total_rows // CHUNKSIZE) + (1 if total_rows % CHUNKSIZE > 0 else 0))
                    logging.info(f"Estimated rows to process: {total_rows} in approx {n_chunks} chunks.")
                else:
                    logging.info("No rows found for the specified date range.")
                    n_chunks = 0
            except Exception as e:
                logging.warning(f"Could not estimate total rows: {e}. Progress bar may be inaccurate.")
                n_chunks = None

            # Read data in chunks
            if total_rows > 0:
                chunk_iterator = pd.read_sql_query(query, conn, params=params, chunksize=CHUNKSIZE)
                for chunk in tqdm(chunk_iterator, total=n_chunks, desc="Loading Data Chunks"):
                    if not chunk.empty:
                        chunk['game_date'] = pd.to_datetime(chunk['game_date'], errors='coerce')
                        chunk.dropna(subset=['game_date'], inplace=True)
                        all_chunks.append(chunk)
            else:
                 logging.warning("Skipping chunk loading as no rows were estimated.")

    except sqlite3.Error as e:
        logging.error(f"SQLite error during data loading: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error during data loading: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    if not all_chunks:
        logging.warning("No data loaded for the specified date range.")
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Successfully loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df

# <<< --- START: Integrated Scoring Logic from MLb_v3.4.2.py --- >>>

def calculate_single_game_score(df_in, is_pitcher):
    """
    Calculates the daily performance score (-10 to 10) based on WPA and role-specific stats
    for a single player-game observation (takes an aggregated DataFrame row/group).
    Ensures required columns ('game_date', 'player_id', 'game_pk') are always returned.
    """
    # Initialize default return structure
    default_return = {
        'daily_score': np.nan,
        'game_date': pd.NaT,
        'player_id': pd.NA,
        'game_pk': pd.NA,
        'daily_fip': np.nan, 'daily_war': np.nan, 'IP': np.nan,
        'daily_woba': np.nan, 'daily_wrc_plus': np.nan, 'daily_owar': np.nan, 'PA': np.nan
    }

    if df_in is None or df_in.empty:
        return pd.DataFrame([default_return]) # Return DataFrame with defaults

    # Ensure we work with a single row's data (or handle multi-row input if needed)
    if len(df_in) > 1:
         # If multiple rows are passed (e.g., from apply), process the first one
         df = df_in.iloc[[0]].copy() # Use iloc[[0]] to keep it as a DataFrame
    else:
         df = df_in.copy()

    # --- Get Identifiers ---
    # Use .get() with default for safety, then extract scalar value
    game_date = df.get('game_date', pd.Series([pd.NaT], index=df.index)).iloc[0]
    player_id = df.get('player_id', pd.Series([pd.NA], index=df.index)).iloc[0]
    game_pk = df.get('game_pk', pd.Series([pd.NA], index=df.index)).iloc[0]

    # --- Common Requirement: WPA ---
    daily_wpa = df.get('daily_wpa', pd.Series([0.0], index=df.index)).iloc[0]
    daily_wpa = safe_get_numeric(pd.Series([daily_wpa])).iloc[0] # Ensure numeric

    # --- Role-Specific Calculations ---
    daily_fip, daily_war, IP = np.nan, np.nan, np.nan
    daily_woba, daily_wrc_plus, daily_owar, PA = np.nan, np.nan, np.nan, np.nan
    daily_score = np.nan # Initialize score

    try: # Add try-except block for robustness within the calculation
        if is_pitcher:
            outs_recorded = df.get('outs_recorded', pd.Series([0], index=df.index)).iloc[0]
            home_run = df.get('home_run', pd.Series([0], index=df.index)).iloc[0]
            walk = df.get('walk', pd.Series([0], index=df.index)).iloc[0]
            hit_by_pitch = df.get('hit_by_pitch', pd.Series([0], index=df.index)).iloc[0]
            strikeout = df.get('strikeout', pd.Series([0], index=df.index)).iloc[0]

            outs_recorded = safe_get_numeric(pd.Series([outs_recorded])).iloc[0]
            home_run = safe_get_numeric(pd.Series([home_run])).iloc[0]
            walk = safe_get_numeric(pd.Series([walk])).iloc[0]
            hit_by_pitch = safe_get_numeric(pd.Series([hit_by_pitch])).iloc[0]
            strikeout = safe_get_numeric(pd.Series([strikeout])).iloc[0]

            IP = outs_recorded / 3.0

            fip_constant = LEAGUE_AVERAGES['FIP_constant']
            fip_numerator = (13 * home_run + 3 * (walk + hit_by_pitch) - 2 * strikeout)
            lg_fip = LEAGUE_AVERAGES['lgFIP']
            penalty_fip = lg_fip + 1.0
            daily_fip = penalty_fip if IP <= 0.001 else (fip_numerator / IP) + fip_constant

            runs_per_win_war = LEAGUE_AVERAGES['runs_per_win']
            daily_war = 0.0
            if runs_per_win_war > 0 and IP > 0.001:
                daily_war = ((lg_fip - daily_fip) / runs_per_win_war) * IP

            scaled_war = daily_war * 15
            scaled_wpa = daily_wpa * 20
            daily_score = 0.7 * scaled_war + 0.3 * scaled_wpa

        else: # Hitters
            PA = df.get('PA', pd.Series([0], index=df.index)).iloc[0]
            woba_value = df.get('woba_value', pd.Series([0.0], index=df.index)).iloc[0]
            PA = safe_get_numeric(pd.Series([PA])).iloc[0]
            woba_value = safe_get_numeric(pd.Series([woba_value])).iloc[0]

            league_woba = LEAGUE_AVERAGES['league_woba']
            woba_scale = LEAGUE_AVERAGES['wOBA_scale']
            runs_per_win_owar = LEAGUE_AVERAGES['runs_per_win']

            daily_woba = 0.0
            daily_wrc_plus = 100.0
            daily_owar = 0.0

            if league_woba > 0 and woba_scale > 0 and runs_per_win_owar > 0:
                if PA > 0:
                    daily_woba = woba_value / PA
                    daily_wrc_plus = 100.0 * (daily_woba / league_woba)
                    wRAA = ((daily_woba - league_woba) / woba_scale) * PA
                    daily_owar = wRAA / runs_per_win_owar
            else:
                 logging.warning("Invalid league constants for hitter score calculation.")

            scaled_wpa = daily_wpa * 20
            scaled_wrc_plus = (daily_wrc_plus - 100) / 10.0
            scaled_owar = daily_owar * 50
            daily_score = 0.45 * scaled_wpa + 0.35 * scaled_wrc_plus + 0.2 * scaled_owar

    except Exception as e:
         logging.error(f"Error calculating score for player {player_id} on {game_date}: {e}")
         daily_score = np.nan # Ensure score is NaN on error

    # --- Final Step: Clamp and Clean Score ---
    final_score = np.clip(daily_score, -10, 10)
    final_score = np.nan_to_num(final_score, nan=0.0) # Use 0 for NaN scores

    # Construct result dictionary ensuring all keys exist
    result_data = {
        'daily_score': final_score,
        'game_date': game_date,
        'player_id': player_id,
        'game_pk': game_pk,
        'daily_fip': daily_fip, 'daily_war': daily_war, 'IP': IP,
        'daily_woba': daily_woba, 'daily_wrc_plus': daily_wrc_plus, 'daily_owar': daily_owar, 'PA': PA
    }

    return pd.DataFrame([result_data]) # Return as DataFrame


def aggregate_and_calculate_scores(df):
    """
    Aggregates pitch-level data to game-player level and calculates daily scores.
    Includes game_pk in the output.
    """
    logging.info(f"Aggregating and calculating scores for {len(df)} rows...")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # --- Pre-aggregation Processing ---
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    for id_col in ['batter', 'pitcher', 'game_pk']: # Ensure game_pk is numeric
        if id_col in df.columns:
            df[id_col] = pd.to_numeric(df[id_col], errors='coerce').astype('Int64')

    df.dropna(subset=['game_date', 'batter', 'pitcher', 'events', 'game_pk'], inplace=True) # Ensure game_pk is not NaN
    if df.empty:
        logging.warning("No valid rows remaining after initial NaN drop (incl. game_pk).")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate PA per Pitch Event
    if 'woba_denom' in df.columns:
        df['is_pa_outcome'] = (safe_get_numeric(df['woba_denom'], 0) == 1).astype(int)
    else:
        logging.warning("Missing 'woba_denom'. Inferring PA from events (less accurate).")
        pa_ending_events = [
            'strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run',
            'field_out', 'sac_fly', 'sac_bunt', 'force_out', 'grounded_into_double_play',
            'fielders_choice', 'field_error', 'double_play', 'triple_play', 'batter_interference',
            'catcher_interf'
        ]
        df['is_pa_outcome'] = df['events'].astype(str).isin(pa_ending_events).astype(int)

    # Calculate Outs per Event
    df['outs_made'] = 0
    if 'events' in df.columns:
        event_out_map = {
            'strikeout': 1, 'field_out': 1, 'sac_fly': 1, 'sac_bunt': 1, 'force_out': 1,
            'grounded_into_double_play': 2, 'double_play': 2, 'triple_play': 3,
            'fielders_choice_out': 1, 'caught_stealing_2b': 1, 'caught_stealing_3b': 1, 'caught_stealing_home': 1,
            'pickoff_1b': 1, 'pickoff_2b': 1, 'pickoff_3b': 1, 'pickoff_caught_stealing_2b': 1,
            'pickoff_caught_stealing_3b': 1, 'pickoff_caught_stealing_home': 1, 'other_out': 1,
            'sac_fly_double_play': 2, 'sac_bunt_double_play': 2, 'strikeout_double_play': 2
        }
        df['outs_made'] = df['events'].astype(str).map(event_out_map).fillna(0)
        if 'description' in df.columns:
            desc_lower = df['description'].astype(str).str.lower()
            df.loc[desc_lower.str.contains('strikes out', na=False), 'outs_made'] = np.maximum(df['outs_made'], 1)
            df.loc[desc_lower.str.contains('double play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 2)
            df.loc[desc_lower.str.contains('triple play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 3)
        df['outs_made'] = df['outs_made'].astype(int)

    # Event Flags
    event_flags = { 'single': 'single', 'double': 'double', 'triple': 'triple', 'home_run': 'home_run',
                    'walk': 'walk', 'hit_by_pitch': 'hit_by_pitch', 'strikeout': 'strikeout'}
    if 'events' in df.columns:
        events_str = df['events'].astype(str)
        for event, flag_col in event_flags.items():
            df[flag_col] = (events_str == event).astype(int)
        if 'description' in df.columns:
            desc_lower = df['description'].astype(str).str.lower()
            df.loc[desc_lower.str.contains('strikes out', na=False), 'strikeout'] = 1
    else:
        for flag_col in event_flags.values(): df[flag_col] = 0

    # Calculate Barrel if missing
    calculate_barrel = False
    if 'is_barrel' not in df.columns:
        calculate_barrel = True
    elif df['is_barrel'].isnull().any():
         df['is_barrel'] = safe_get_numeric(df['is_barrel'], 0)
         if df['is_barrel'].isnull().any():
             calculate_barrel = True
    if calculate_barrel:
        if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
            ls = safe_get_numeric(df['launch_speed'], 0.0)
            la = safe_get_numeric(df['launch_angle'], 0.0)
            ls_cond = ls >= 98
            la_min = np.select([ls >= 116, ls >= 110, ls >= 105, ls >= 100, ls >= 98], [8.0, 12.0, 16.0, 20.0, 26.0], default=-np.inf)
            la_max = np.select([ls >= 116, ls >= 110, ls >= 105, ls >= 100, ls >= 98], [33.0, 36.0, 39.0, 44.0, 50.0], default=np.inf)
            la_cond = (la >= la_min) & (la <= la_max)
            barrel_mask = (ls >= 98) & la_cond
            df['is_barrel'] = barrel_mask.astype(int)
        else:
            df['is_barrel'] = 0
    else:
        df['is_barrel'] = df['is_barrel'].astype(int)

    # Numeric Conversions & Default Fills
    numeric_cols_agg = ['delta_home_win_exp', 'woba_value', 'release_speed', 'release_spin_rate',
                        'launch_speed', 'launch_angle', 'pfx_x', 'pfx_z']
    for col in numeric_cols_agg:
        if col in df.columns:
            default_val = LEAGUE_AVERAGES.get(col, 0.0)
            df[col] = safe_get_numeric(df[col], default_val)

    # --- Aggregation ---
    logging.info("Starting aggregation...")
    # Group by game_pk as well to retain game context
    group_cols_batter = ['game_date', 'game_pk', 'batter']
    group_cols_pitcher = ['game_date', 'game_pk', 'pitcher']

    agg_ops = {
        'single': 'sum', 'double': 'sum', 'triple': 'sum', 'home_run': 'sum',
        'walk': 'sum', 'hit_by_pitch': 'sum', 'strikeout': 'sum',
        'is_barrel': 'sum', 'woba_value': 'sum', 'delta_home_win_exp': 'sum',
        'is_pa_outcome': 'sum', 'outs_made': 'sum',
        'release_speed': 'mean', 'release_spin_rate': 'mean',
        'launch_speed': 'mean', 'launch_angle': 'mean',
        'pfx_x': 'mean', 'pfx_z': 'mean',
    }
    valid_agg_ops = {k: v for k, v in agg_ops.items() if k in df.columns}
    if not valid_agg_ops:
        logging.error("No valid columns found for aggregation.")
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate for Batters
    batter_daily_stats = pd.DataFrame()
    if 'batter' in df.columns:
        try:
            batter_agg = df.groupby(group_cols_batter, as_index=False, observed=True).agg(valid_agg_ops)
            batter_agg.rename(columns={'is_pa_outcome': 'PA', 'delta_home_win_exp': 'daily_wpa', 'batter': 'player_id'}, inplace=True)
            batter_agg.dropna(subset=['player_id'], inplace=True)
            batter_agg['player_id'] = batter_agg['player_id'].astype(int)
            logging.info(f"Aggregated {len(batter_agg)} batter-game rows.")
            if not batter_agg.empty:
                 # Use apply to process each group (player-game)
                 # Note: Using apply can be slow on large datasets. Consider vectorization if performance is critical.
                 results = []
                 for name, group in tqdm(batter_agg.groupby(['game_date', 'player_id', 'game_pk'], observed=True), desc="Scoring Batters"):
                      results.append(calculate_single_game_score(group, is_pitcher=False))

                 if results:
                      batter_daily_stats = pd.concat(results, ignore_index=True)
                 else:
                      logging.warning("No batter scores generated.")
            else:
                 logging.warning("Batter aggregation resulted in empty DataFrame.")
        except Exception as e:
             logging.error(f"Error during batter aggregation/scoring: {e}")
             traceback.print_exc()

    # Aggregate for Pitchers
    pitcher_daily_stats = pd.DataFrame()
    if 'pitcher' in df.columns:
        try:
            pitcher_agg = df.groupby(group_cols_pitcher, as_index=False, observed=True).agg(valid_agg_ops)
            pitcher_agg.rename(columns={'outs_made': 'outs_recorded', 'delta_home_win_exp': 'daily_wpa', 'pitcher': 'player_id'}, inplace=True)
            pitcher_agg.dropna(subset=['player_id'], inplace=True)
            pitcher_agg['player_id'] = pitcher_agg['player_id'].astype(int)
            logging.info(f"Aggregated {len(pitcher_agg)} pitcher-game rows.")
            if not pitcher_agg.empty:
                 # Use apply to process each group (player-game)
                 results = []
                 for name, group in tqdm(pitcher_agg.groupby(['game_date', 'player_id', 'game_pk'], observed=True), desc="Scoring Pitchers"):
                      results.append(calculate_single_game_score(group, is_pitcher=True))

                 if results:
                      pitcher_daily_stats = pd.concat(results, ignore_index=True)
                 else:
                      logging.warning("No pitcher scores generated.")
            else:
                 logging.warning("Pitcher aggregation resulted in empty DataFrame.")
        except Exception as e:
             logging.error(f"Error during pitcher aggregation/scoring: {e}")
             traceback.print_exc()

    logging.info(f"Finished score calculation. Batters: {len(batter_daily_stats)}, Pitchers: {len(pitcher_daily_stats)}")
    # Ensure correct dtypes before returning
    if not batter_daily_stats.empty:
         batter_daily_stats['game_date'] = pd.to_datetime(batter_daily_stats['game_date'])
         batter_daily_stats['player_id'] = batter_daily_stats['player_id'].astype(int)
         batter_daily_stats['game_pk'] = pd.to_numeric(batter_daily_stats['game_pk'], errors='coerce').astype('Int64') # Keep game_pk nullable
    if not pitcher_daily_stats.empty:
         pitcher_daily_stats['game_date'] = pd.to_datetime(pitcher_daily_stats['game_date'])
         pitcher_daily_stats['player_id'] = pitcher_daily_stats['player_id'].astype(int)
         pitcher_daily_stats['game_pk'] = pd.to_numeric(pitcher_daily_stats['game_pk'], errors='coerce').astype('Int64') # Keep game_pk nullable

    return batter_daily_stats, pitcher_daily_stats

# <<< --- END: Integrated Scoring Logic --- >>>


def calculate_rolling_stats(daily_scores_df):
    """
    Calculates rolling performance percentiles and averages for each player.
    Expects 'game_pk' column to be present.
    """
    logging.info("Calculating rolling statistics...")
    if daily_scores_df.empty or 'daily_score' not in daily_scores_df.columns:
        logging.warning("Daily scores DataFrame is empty or missing 'daily_score'. Skipping rolling stats.")
        return daily_scores_df
    if 'game_pk' not in daily_scores_df.columns:
         logging.error("Missing 'game_pk' column in daily scores. Cannot calculate rolling stats accurately.")
         return pd.DataFrame()


    # Ensure player_id is numeric before sorting/grouping
    daily_scores_df['player_id'] = pd.to_numeric(daily_scores_df['player_id'], errors='coerce')
    daily_scores_df.dropna(subset=['player_id'], inplace=True)
    daily_scores_df['player_id'] = daily_scores_df['player_id'].astype(int)
    # Ensure game_date is datetime
    daily_scores_df['game_date'] = pd.to_datetime(daily_scores_df['game_date'])

    # Sort by player and date (essential for rolling calculations)
    # Include game_pk in sort in case of doubleheaders, although aggregation already grouped by it
    df = daily_scores_df.sort_values(by=['player_id', 'game_date', 'game_pk']).copy()
    df.set_index('game_date', inplace=True) # Set date index for rolling time window

    window_str = f'{ROLLING_WINDOW_DAYS}D'
    min_periods_roll = max(1, ROLLING_WINDOW_DAYS // 3)
    min_periods_median = max(1, ROLLING_WINDOW_DAYS // 2)

    # Group by player_id to calculate rolling stats per player
    grouped = df.groupby('player_id')

    # Calculate rolling percentile
    logging.info(f"Calculating rolling percentile with window='{window_str}', min_periods={min_periods_roll}")
    df[f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile'] = grouped['daily_score'].transform(
        lambda x: x.rolling(window=window_str, min_periods=min_periods_roll, closed='left')
                 .apply(lambda y: pd.Series(y).rank(pct=True).iloc[-1] * 100 if (y.size > 0 and not pd.isna(y.iloc[-1])) else np.nan, raw=False)
    )

    # Calculate rolling mean
    logging.info(f"Calculating rolling mean with window='{window_str}', min_periods={min_periods_roll}")
    df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean'] = grouped[ROLLING_AVG_METRIC].transform(
        lambda x: x.rolling(window=window_str, min_periods=min_periods_roll, closed='left').mean()
    )

    # Calculate rolling median threshold
    if ROLLING_AVG_THRESHOLD_TYPE == 'player_median':
        logging.info(f"Calculating rolling median threshold with window='{window_str}', min_periods={min_periods_median}")
        df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'] = grouped[ROLLING_AVG_METRIC].transform(
             lambda x: x.rolling(window=window_str, min_periods=min_periods_median, closed='left').median()
        )
    else:
        logging.warning(f"Unsupported ROLLING_AVG_THRESHOLD_TYPE: {ROLLING_AVG_THRESHOLD_TYPE}. Threshold column not created.")
        df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'] = np.nan

    # Fill NaNs in rolling features
    roll_cols = [col for col in df.columns if col.startswith('roll_')]
    for col in roll_cols:
        df[col].fillna(0, inplace=True)

    logging.info("Finished calculating rolling statistics.")
    df.reset_index(inplace=True) # Reset index back to default
    return df


def identify_breakouts(player_perf_df):
    """
    Filters player performance data to identify potential breakout games,
    then selects the top N per day based on game count.
    """
    logging.info("Identifying potential breakout performances...")
    if player_perf_df.empty:
        logging.warning("Input DataFrame is empty. Cannot identify breakouts.")
        return pd.DataFrame()
    if 'game_pk' not in player_perf_df.columns:
         logging.error("Missing 'game_pk' column in performance history. Cannot count games per day.")
         return pd.DataFrame()

    # Ensure player_id is integer type before filtering
    player_perf_df['player_id'] = pd.to_numeric(player_perf_df['player_id'], errors='coerce').astype('Int64')
    player_perf_df.dropna(subset=['player_id'], inplace=True)
    player_perf_df['player_id'] = player_perf_df['player_id'].astype(int)

    # --- Initial Filtering ---
    required_cols = [
        'player_id', 'daily_score', 'game_date', 'game_pk',
        f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile',
        f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean',
    ]
    threshold_col = f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'
    if threshold_col in player_perf_df.columns:
        required_cols.append(threshold_col)
    else:
        logging.warning(f"Threshold column '{threshold_col}' not found. Rolling average filter might be skipped or default to False.")

    missing_req = [col for col in required_cols if col not in player_perf_df.columns]
    if missing_req:
        logging.error(f"Missing required columns for breakout identification: {missing_req}. Cannot proceed.")
        return pd.DataFrame()

    df = player_perf_df.copy()

    # 1. Performance Percentile Filter
    percentile_cond = df[f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile'] > PERFORMANCE_PERCENTILE_THRESHOLD

    # 2. Rolling Average Filter (Below Threshold)
    rolling_avg_cond = pd.Series(False, index=df.index) # Default to False
    if threshold_col in df.columns:
         mean_col_numeric = safe_get_numeric(df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean'])
         threshold_col_numeric = safe_get_numeric(df[threshold_col])
         valid_comparison = mean_col_numeric.notna() & threshold_col_numeric.notna()
         rolling_avg_cond[valid_comparison] = mean_col_numeric[valid_comparison] < threshold_col_numeric[valid_comparison]
    else:
         logging.warning(f"Skipping rolling average filter as threshold column '{threshold_col}' is missing.")

    # 3. Non-Superstar Filter
    superstar_cond = ~df['player_id'].isin(SUPERSTAR_PLAYER_IDS)

    # Combine initial filters
    initial_candidates = df[percentile_cond & rolling_avg_cond & superstar_cond].copy()
    logging.info(f"Identified {len(initial_candidates)} initial breakout candidates.")

    if initial_candidates.empty:
        return pd.DataFrame()

    # --- Daily Top N Selection ---
    logging.info("Selecting Top N breakouts per day based on game count...")

    # Ensure game_pk is suitable for nunique()
    initial_candidates['game_pk'] = pd.to_numeric(initial_candidates['game_pk'], errors='coerce')

    def select_top_n_for_date(date_group):
        """Applied to each date group to select top N."""
        # Count unique valid game_pk values for the date
        # Drop NaN game_pks before counting unique values
        num_games = date_group['game_pk'].dropna().nunique()

        if num_games >= GAMES_THRESHOLD:
            n_to_select = TOP_N_HIGH_GAMES
        else:
            n_to_select = TOP_N_LOW_GAMES

        # Sort candidates within the day by score and take top N
        return date_group.sort_values(by='daily_score', ascending=False).head(n_to_select)

    # Apply the selection function to each date group
    # Ensure game_date is datetime before grouping
    initial_candidates['game_date'] = pd.to_datetime(initial_candidates['game_date'])
    final_breakouts = initial_candidates.groupby(initial_candidates['game_date'].dt.date, group_keys=False).apply(select_top_n_for_date)

    logging.info(f"Selected {len(final_breakouts)} final breakout performances after daily Top N filtering.")
    return final_breakouts


def enrich_breakouts(breakout_df, pitch_data_df):
    """
    Adds contextual information (weather, ballpark, opponent) to breakout performances.
    Placeholder - Requires implementation for fetching/merging context.
    """
    logging.info("Enriching breakout performances with context (Placeholder)...")
    if breakout_df.empty:
        logging.warning("Breakout DataFrame is empty. No enrichment needed.")
        return breakout_df

    enriched_df = breakout_df.copy()

    # --- Add Context Columns (Initialize with defaults) ---
    enriched_df['weather_temp'] = np.nan
    enriched_df['weather_wind_speed'] = np.nan
    enriched_df['weather_wind_dir'] = np.nan
    enriched_df['weather_humidity'] = np.nan
    enriched_df['weather_pressure'] = np.nan
    enriched_df['ballpark'] = 'Unknown'
    enriched_df['opponent_team'] = 'Unknown'
    enriched_df['opponent_pitcher'] = pd.NA # Use pandas NA for nullable integer
    enriched_df['opponent_lineup_strength'] = np.nan # For pitchers (example)

    # TODO: Implement enrichment logic:
    # Needs access to game-level info (home/away teams, etc.) associated with the game_pk
    # This might require merging `enriched_df` back with a game-level summary derived from the original pitch data.
    logging.warning("Enrichment logic is not yet implemented.")
    return enriched_df


def save_breakouts(breakout_df):
    """Saves the identified and enriched breakout performances to a database."""
    logging.info(f"Saving {len(breakout_df)} breakout performances to {OUTPUT_DB_PATH} table '{OUTPUT_TABLE_NAME}'...")
    if breakout_df.empty:
        logging.warning("No breakout data to save.")
        return

    try:
        with sqlite3.connect(OUTPUT_DB_PATH) as conn:
            df_to_save = breakout_df.copy()
            for col in df_to_save.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
                df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d')

            if 'player_id' in df_to_save.columns:
                 df_to_save['player_id'] = pd.to_numeric(df_to_save['player_id'], errors='coerce').astype('Int64')
            if 'game_pk' in df_to_save.columns:
                 df_to_save['game_pk'] = pd.to_numeric(df_to_save['game_pk'], errors='coerce').astype('Int64')


            # Ensure numeric columns don't have Pandas NA before saving
            for col in df_to_save.select_dtypes(include=[np.number]).columns:
                 if df_to_save[col].isnull().any():
                      if pd.api.types.is_integer_dtype(df_to_save[col]):
                           # For nullable integers that might still have <NA>, convert to float then fill
                           df_to_save[col] = df_to_save[col].astype(float)
                      df_to_save[col].fillna(np.nan, inplace=True) # Use np.nan for database NULL

            # Use 'replace' to overwrite the table each time this script runs
            df_to_save.to_sql(OUTPUT_TABLE_NAME, conn, if_exists='replace', index=False)
            logging.info(f"Successfully saved data to table '{OUTPUT_TABLE_NAME}'.")
    except sqlite3.Error as e:
        logging.error(f"SQLite error saving breakout data: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving breakout data: {e}")
        traceback.print_exc()


# === Main Execution ===
if __name__ == "__main__":
    logging.info("--- Starting Breakout Scout ---")
    start_time = time.time()

    # 1. Load historical pitch data
    raw_pitch_data = load_historical_data(ANALYSIS_START_DATE, ANALYSIS_END_DATE)

    if not raw_pitch_data.empty:
        # 2. Aggregate pitch data and calculate daily scores
        batters_daily, pitchers_daily = aggregate_and_calculate_scores(raw_pitch_data)
        # Explicitly delete large raw data frame to free memory
        del raw_pitch_data
        gc.collect()
        logging.info("Raw pitch data deleted from memory.")


        # Combine batter and pitcher data for rolling stats calculation
        valid_dfs = []
        if not batters_daily.empty and 'player_id' in batters_daily.columns:
            valid_dfs.append(batters_daily)
        if not pitchers_daily.empty and 'player_id' in pitchers_daily.columns:
             valid_dfs.append(pitchers_daily)

        if valid_dfs:
            all_daily_scores = pd.concat(valid_dfs, ignore_index=True)
            del batters_daily, pitchers_daily, valid_dfs
            gc.collect()
            logging.info("Intermediate daily score DataFrames deleted.")


            if not all_daily_scores.empty:
                # 3. Calculate rolling stats
                player_performance_history = calculate_rolling_stats(all_daily_scores)
                del all_daily_scores
                gc.collect()

                # 4. Identify potential breakouts and select Top N per day
                breakout_candidates = identify_breakouts(player_performance_history)
                del player_performance_history
                gc.collect()


                if not breakout_candidates.empty:
                    # 5. Enrich breakouts with context (Placeholder)
                    enriched_breakouts = enrich_breakouts(breakout_candidates, pd.DataFrame()) # Pass empty df for now

                    # 6. Save results
                    save_breakouts(enriched_breakouts)
                else:
                    logging.info("No breakout candidates identified after filtering.")
            else:
                 logging.warning("Combined daily scores DataFrame is empty, cannot calculate rolling stats.")
        else:
            logging.warning("No valid daily scores calculated (batters or pitchers), cannot proceed.")
    else:
        logging.warning("No pitch data loaded, cannot proceed.")

    end_time = time.time()
    logging.info(f"--- Breakout Scout Finished ---")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

