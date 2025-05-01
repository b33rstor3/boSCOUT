# breakout_scout.py
# Module to identify historical breakout performances from MLB data.

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
from tqdm import tqdm # <--- Added import for progress bar

# Attempt to import meteostat for historical weather
try:
    from meteostat import Point, Daily
except ImportError:
    logging.warning("meteostat library not found. Historical weather features will use defaults or require manual addition. Install with: pip install meteostat")
    Point, Daily = None, None

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# === Configuration & Constants ===
DB_PATH = "mlb_data2.db" # Path to the SQLite database
TABLE_NAME = "pitch_data" # Table containing pitch-by-pitch data
OUTPUT_DB_PATH = "breakouts.db" # Database to store identified breakouts
OUTPUT_TABLE_NAME = "breakout_performances"
LOG_FILE = "breakout_scout.log"

# Date range for historical analysis (adjust as needed)
# Example: Analyze the 2023 season
ANALYSIS_START_DATE = "2023-03-30"
ANALYSIS_END_DATE = "2023-10-01"

CHUNKSIZE = 10000 # Process data in chunks to manage memory

# Breakout Definition Parameters
ROLLING_WINDOW_DAYS = 30 # Look back period for rolling stats/percentiles
PERFORMANCE_PERCENTILE_THRESHOLD = 95 # Performance must be > this percentile of player's recent history
ROLLING_AVG_METRIC = 'daily_score' # Metric to use for rolling average comparison
# TODO: Define the 'below median' threshold more precisely. Is it median of all players, or player's own median?
ROLLING_AVG_THRESHOLD_TYPE = 'player_median' # Options: 'player_median', 'league_median', 'fixed_value'

# TODO: Define the list of superstar player IDs to exclude, or a method to determine them dynamically.
SUPERSTAR_PLAYER_IDS = {
    123456, # Example: Aaron Judge ID
    654321, # Example: Shohei Ohtani ID
    # Add more known superstar IDs here
}

# League averages (can be loaded dynamically or use placeholders)
# Copied from MLb_v3.4.2.py for now, might need refinement/dynamic calculation
LEAGUE_AVERAGES = {
    'release_speed': 93.0, 'spin_rate': 2400.0, 'launch_speed': 88.0,
    'launch_angle': 12.0, 'estimated_ba_using_speedangle': 0.245,
    'estimated_woba_using_speedangle': 0.315, 'pfx_x': 0.0, 'pfx_z': 0.5,
    'temp': 70.0, 'humidity': 50.0, 'wind_speed': 5.0, 'wind_dir': 0.0,
    'pressure': 1013.25, 'league_woba': 0.315, 'wOBA_scale': 1.2,
    'FIP_constant': 3.10, 'lgFIP': 3.10, 'runs_per_win': 9.0
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
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# === Helper Functions ===
def safe_get_numeric(series, default=0.0):
    """Convert series to numeric, coercing errors and filling NaNs with a default."""
    if series is None or series.empty:
        return series
    return pd.to_numeric(series, errors='coerce').fillna(default)

# === Core Logic Functions ===

def load_historical_data(start_date_str, end_date_str):
    """
    Loads pitch data from the SQLite database for the specified date range using chunking.
    """
    logging.info(f"Loading pitch data from {start_date_str} to {end_date_str} from {DB_PATH}")
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found: {DB_PATH}")
        return pd.DataFrame()

    all_chunks = []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Construct the query dynamically based on available columns later if needed
            # For now, assume necessary columns exist as per MLb_v3.4.2.py
            query = f"SELECT * FROM {TABLE_NAME} WHERE game_date BETWEEN ? AND ?"
            params = (start_date_str, end_date_str)

            # Estimate total rows for progress bar (optional but helpful)
            total_rows = 0
            n_chunks = None # Initialize n_chunks
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
                    n_chunks = 0 # Set to 0 if no rows
            except Exception as e:
                logging.warning(f"Could not estimate total rows: {e}. Progress bar may be inaccurate.")
                n_chunks = None # Disable progress bar total if count fails

            # Read data in chunks only if there are rows to process
            if total_rows > 0:
                chunk_iterator = pd.read_sql_query(query, conn, params=params, chunksize=CHUNKSIZE)

                # Use tqdm for the progress bar
                for chunk in tqdm(chunk_iterator, total=n_chunks, desc="Loading Data Chunks"):
                    if not chunk.empty:
                        # Basic type conversion (ensure game_date is datetime)
                        chunk['game_date'] = pd.to_datetime(chunk['game_date'], errors='coerce')
                        chunk.dropna(subset=['game_date'], inplace=True) # Drop rows where date conversion failed
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

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Successfully loaded {len(df)} rows.")
    return df

def calculate_daily_scores(df):
    """
    Aggregates pitch data to daily stats and calculates performance scores.
    Placeholder - Requires the scoring logic from MLb_v3.4.2.py.
    """
    logging.info("Calculating daily scores (Placeholder - Needs full implementation)...")
    # TODO: Integrate the full calculate_daily_score function from MLb_v3.4.2.py
    # This involves:
    # 1. Aggregating pitch-level data to game-level stats per player (PA, outs, woba_value, delta_home_win_exp, etc.)
    # 2. Calculating FIP, WAR for pitchers.
    # 3. Calculating wOBA, wRC+, oWAR for batters.
    # 4. Combining metrics into a scaled 'daily_score' from -10 to 10.
    # 5. Separating batters and pitchers.

    # Placeholder aggregation and score calculation:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Example: Group by game and player, calculate a dummy score
    id_cols = ['game_date', 'batter', 'pitcher']
    agg_cols = [col for col in id_cols if col in df.columns]
    if not agg_cols:
        logging.error("Missing essential ID columns (game_date, batter, pitcher) for aggregation.")
        return pd.DataFrame(), pd.DataFrame()

    # Minimal aggregation example
    agg_dict = {'delta_home_win_exp': 'sum'} # Need more stats for real scoring
    valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    if not valid_agg_dict:
         logging.warning("No valid columns found for placeholder aggregation.")
         # Create dummy score column if needed downstream
         df['daily_score'] = np.random.uniform(-5, 5, size=len(df))
         # Split based on pitch_type (assuming it exists)
         if 'pitch_type' in df.columns:
             batters_daily = df[df['pitch_type'].isnull()][['game_date', 'batter', 'daily_score']].dropna(subset=['batter']).rename(columns={'batter': 'player_id'})
             pitchers_daily = df[df['pitch_type'].notnull()][['game_date', 'pitcher', 'daily_score']].dropna(subset=['pitcher']).rename(columns={'pitcher': 'player_id'})
             # Ensure player_id is integer type after dropping NaNs
             batters_daily['player_id'] = pd.to_numeric(batters_daily['player_id'], errors='coerce').astype('Int64')
             pitchers_daily['player_id'] = pd.to_numeric(pitchers_daily['player_id'], errors='coerce').astype('Int64')
             batters_daily.dropna(subset=['player_id'], inplace=True)
             pitchers_daily.dropna(subset=['player_id'], inplace=True)
             batters_daily['player_id'] = batters_daily['player_id'].astype(int)
             pitchers_daily['player_id'] = pitchers_daily['player_id'].astype(int)
             return batters_daily, pitchers_daily
         else:
              logging.error("Missing 'pitch_type' column, cannot separate batters/pitchers.")
              return pd.DataFrame(), pd.DataFrame()


    # --- This section needs the full aggregation and scoring logic ---
    logging.warning("Using placeholder scoring logic. Results will not be accurate.")
    # Example split (replace with proper logic)
    batters_daily = df[df['pitch_type'].isnull()].copy()
    pitchers_daily = df[df['pitch_type'].notnull()].copy()
    batters_daily['daily_score'] = np.random.uniform(-5, 5, size=len(batters_daily))
    pitchers_daily['daily_score'] = np.random.uniform(-5, 5, size=len(pitchers_daily))

    # Select and rename columns
    batters_out = batters_daily[['game_date', 'batter', 'daily_score']].rename(columns={'batter': 'player_id'}).dropna(subset=['player_id'])
    pitchers_out = pitchers_daily[['game_date', 'pitcher', 'daily_score']].rename(columns={'pitcher': 'player_id'}).dropna(subset=['player_id'])
    # Ensure player_id is integer type after dropping NaNs
    batters_out['player_id'] = pd.to_numeric(batters_out['player_id'], errors='coerce').astype('Int64')
    pitchers_out['player_id'] = pd.to_numeric(pitchers_out['player_id'], errors='coerce').astype('Int64')
    batters_out.dropna(subset=['player_id'], inplace=True)
    pitchers_out.dropna(subset=['player_id'], inplace=True)
    batters_out['player_id'] = batters_out['player_id'].astype(int)
    pitchers_out['player_id'] = pitchers_out['player_id'].astype(int)
    # --- End Placeholder ---

    logging.info(f"Finished placeholder score calculation. Batters: {len(batters_out)}, Pitchers: {len(pitchers_out)}")
    return batters_out, pitchers_out


def calculate_rolling_stats(daily_scores_df):
    """
    Calculates rolling performance percentiles and averages for each player.
    """
    logging.info("Calculating rolling statistics...")
    if daily_scores_df.empty or 'daily_score' not in daily_scores_df.columns:
        logging.warning("Daily scores DataFrame is empty or missing 'daily_score'. Skipping rolling stats.")
        return daily_scores_df

    # Ensure player_id is numeric before sorting/grouping
    daily_scores_df['player_id'] = pd.to_numeric(daily_scores_df['player_id'], errors='coerce')
    daily_scores_df.dropna(subset=['player_id'], inplace=True)
    daily_scores_df['player_id'] = daily_scores_df['player_id'].astype(int)

    df = daily_scores_df.sort_values(by=['player_id', 'game_date']).copy()

    # Calculate rolling percentile of 'daily_score' over the window
    # shift(1) uses data *before* the current game day
    # Use raw=False as apply function needs Series object
    df[f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile'] = df.groupby('player_id')['daily_score'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_DAYS, min_periods=max(1, ROLLING_WINDOW_DAYS // 3))
                 .apply(lambda y: pd.Series(y).rank(pct=True).iloc[-1] * 100 if not pd.isna(y.iloc[-1]) else np.nan, raw=False)
                 .shift(1)
    )

    # Calculate rolling average of the chosen metric (e.g., 'daily_score')
    # shift(1) uses data *before* the current game day
    df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean'] = df.groupby('player_id')[ROLLING_AVG_METRIC].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_DAYS, min_periods=max(1, ROLLING_WINDOW_DAYS // 3)).mean().shift(1)
    )

    # Calculate the threshold for the rolling average comparison (e.g., player's own median over the window)
    if ROLLING_AVG_THRESHOLD_TYPE == 'player_median':
        df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'] = df.groupby('player_id')[ROLLING_AVG_METRIC].transform(
             lambda x: x.rolling(window=ROLLING_WINDOW_DAYS, min_periods=max(1, ROLLING_WINDOW_DAYS // 2)).median().shift(1) # Use median before the game
        )
    # TODO: Implement 'league_median' or 'fixed_value' threshold types if needed
    else:
        logging.warning(f"Unsupported ROLLING_AVG_THRESHOLD_TYPE: {ROLLING_AVG_THRESHOLD_TYPE}. Threshold column not created.")
        df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'] = np.nan # Add placeholder column


    # Fill NaNs in rolling features (e.g., with 0 or a neutral value)
    roll_cols = [col for col in df.columns if col.startswith('roll_')]
    for col in roll_cols:
        df[col].fillna(0, inplace=True) # Or use a more sophisticated fill based on context

    logging.info("Finished calculating rolling statistics.")
    return df


def identify_breakouts(player_perf_df):
    """
    Filters player performance data to identify potential breakout games.
    """
    logging.info("Identifying potential breakout performances...")
    if player_perf_df.empty:
        logging.warning("Input DataFrame is empty. Cannot identify breakouts.")
        return pd.DataFrame()

    # Ensure player_id is integer type before filtering
    player_perf_df['player_id'] = pd.to_numeric(player_perf_df['player_id'], errors='coerce').astype('Int64')
    player_perf_df.dropna(subset=['player_id'], inplace=True)
    player_perf_df['player_id'] = player_perf_df['player_id'].astype(int)


    required_cols = [
        'player_id', 'daily_score',
        f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile',
        f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean',
        # Check if threshold column exists, handle potential absence
        # f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_median_threshold'
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

    # --- Apply Breakout Filters ---
    # 1. Performance Percentile Filter
    percentile_cond = df[f'roll_{ROLLING_WINDOW_DAYS}d_score_percentile'] > PERFORMANCE_PERCENTILE_THRESHOLD

    # 2. Rolling Average Filter (Below Threshold)
    rolling_avg_cond = pd.Series(False, index=df.index) # Default to False
    if threshold_col in df.columns:
         # Ensure both columns are numeric before comparison
         mean_col_numeric = safe_get_numeric(df[f'roll_{ROLLING_WINDOW_DAYS}d_{ROLLING_AVG_METRIC}_mean'])
         threshold_col_numeric = safe_get_numeric(df[threshold_col])
         # Perform comparison only where both are valid numbers
         valid_comparison = mean_col_numeric.notna() & threshold_col_numeric.notna()
         rolling_avg_cond[valid_comparison] = mean_col_numeric[valid_comparison] < threshold_col_numeric[valid_comparison]
         # Handle cases where threshold might be NaN (e.g., beginning of player history) - already handled by fillna in calculate_rolling_stats
         # rolling_avg_cond = rolling_avg_cond.fillna(False) # Re-apply fillna just in case
    else:
         logging.warning(f"Skipping rolling average filter as threshold column '{threshold_col}' is missing.")


    # 3. Non-Superstar Filter
    superstar_cond = ~df['player_id'].isin(SUPERSTAR_PLAYER_IDS)

    # Combine conditions
    breakout_candidates = df[percentile_cond & rolling_avg_cond & superstar_cond].copy() # Use .copy()

    logging.info(f"Identified {len(breakout_candidates)} potential breakout performances.")
    return breakout_candidates


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
    # 1. Get Game Info: For each breakout (player_id, game_date), find the corresponding game_pk from pitch_data_df.
    #    Need to merge breakout_df with relevant parts of pitch_data_df (like game_pk, home_team, away_team, pitcher/batter IDs).
    #    Handle cases where a player might have played multiple games on the same date if necessary.
    # 2. Get Opponent: Identify the opposing team based on home/away status.
    # 3. Get Ballpark: Identify the home team and map to BALLPARK_COORDS.
    # 4. Fetch Historical Weather: Use Meteostat (or another source) with ballpark coordinates and game_date.
    #    - Requires careful handling of dates and potential missing weather data.
    #    - Cache weather lookups to avoid redundant API calls.
    # 5. Add Opponent Details: Identify opposing starting pitcher (for batters) or estimate lineup strength (for pitchers). This might require more data joins.

    logging.warning("Enrichment logic is not yet implemented.")
    return enriched_df


def save_breakouts(breakout_df):
    """Saves the identified and enriched breakout performances to a database."""
    logging.info(f"Saving {len(breakout_df)} breakout performances to {OUTPUT_DB_PATH}...")
    if breakout_df.empty:
        logging.warning("No breakout data to save.")
        return

    try:
        with sqlite3.connect(OUTPUT_DB_PATH) as conn:
            # Ensure table exists, create if not
            # Convert datetime columns to string for SQLite compatibility if necessary
            df_to_save = breakout_df.copy()
            for col in df_to_save.select_dtypes(include=['datetime64[ns]']).columns:
                df_to_save[col] = df_to_save[col].dt.strftime('%Y-%m-%d') # Store dates as strings

            # Ensure player_id is standard int before saving if needed, handle potential NA
            if 'player_id' in df_to_save.columns:
                 df_to_save['player_id'] = pd.to_numeric(df_to_save['player_id'], errors='coerce').astype('Int64')

            df_to_save.to_sql(OUTPUT_TABLE_NAME, conn, if_exists='replace', index=False) # Use 'replace' for initial runs, 'append' later
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
        # 2. Calculate daily scores (Placeholder)
        batters_daily, pitchers_daily = calculate_daily_scores(raw_pitch_data)

        # Combine batter and pitcher data for rolling stats calculation
        # Ensure player_id columns exist before concatenating
        valid_dfs = []
        if not batters_daily.empty and 'player_id' in batters_daily.columns:
            valid_dfs.append(batters_daily)
        if not pitchers_daily.empty and 'player_id' in pitchers_daily.columns:
             valid_dfs.append(pitchers_daily)

        if valid_dfs:
            all_daily_scores = pd.concat(valid_dfs, ignore_index=True)

            if not all_daily_scores.empty:
                # 3. Calculate rolling stats
                player_performance_history = calculate_rolling_stats(all_daily_scores)

                # 4. Identify potential breakouts
                breakout_candidates = identify_breakouts(player_performance_history)

                if not breakout_candidates.empty:
                    # 5. Enrich breakouts with context (Placeholder)
                    enriched_breakouts = enrich_breakouts(breakout_candidates, raw_pitch_data)

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
