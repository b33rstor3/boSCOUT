import sqlite3
import pandas as pd
import time
from datetime import date, timedelta, datetime
import logging
import traceback
import sys

# Attempt to import pybaseball, provide instructions if missing
try:
    from pybaseball import statcast
except ImportError:
    print("ERROR: pybaseball library not found.")
    print("Please install it by running: pip install pybaseball")
    sys.exit(1)

# --- Configuration ---
DB_PATH = "mlb_data2.db"
TABLE_NAME = "pitch_data"
# ** Adjust these dates as needed **
# Dates are inclusive. Fetch data from March 1st up to (and including) April 17th.
FETCH_START_DATE = "2025-03-01"
FETCH_END_DATE = "2025-04-17" # Day before the last failed prediction date
# Fetch data in smaller chunks to avoid timeouts and be polite to the API
CHUNK_DAYS = 3
# Delay between fetching chunks (in seconds)
FETCH_DELAY = 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_target_columns(db_path, table_name):
    """Gets the list of column names from the specified database table."""
    logging.info(f"Getting column names from table '{table_name}' in {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [info[1] for info in cursor.fetchall()]
            logging.info(f"Found {len(columns)} columns in '{table_name}'.")
            return columns
    except sqlite3.Error as e:
        logging.error(f"SQLite error getting columns for '{table_name}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error getting columns: {e}")
        return None

def fetch_statcast_data_chunk(start_dt_str, end_dt_str):
    """Fetches Statcast data for a specific date range using pybaseball."""
    logging.info(f"Fetching Statcast data from {start_dt_str} to {end_dt_str}...")
    try:
        # pybaseball statcast function retrieves the data
        data = statcast(start_dt=start_dt_str, end_dt=end_dt_str)
        if data is None or data.empty:
            logging.warning(f"No data returned from pybaseball for {start_dt_str} to {end_dt_str}.")
            return pd.DataFrame() # Return empty DataFrame if no data
        logging.info(f"Successfully fetched {len(data)} rows for {start_dt_str} to {end_dt_str}.")
        return data
    except Exception as e:
        logging.error(f"Error fetching Statcast data for {start_dt_str} to {end_dt_str}: {e}")
        logging.error(traceback.format_exc()) # Log full traceback for fetch errors
        return pd.DataFrame() # Return empty DataFrame on error


def clean_and_prepare_data(df, target_columns):
    """Cleans and selects columns from fetched data to match the target table."""
    logging.info(f"Cleaning and preparing {len(df)} fetched rows...")
    if df.empty:
        return df

    # Identify columns present in both fetched data and target table
    cols_to_keep = [col for col in target_columns if col in df.columns]
    missing_cols = [col for col in target_columns if col not in df.columns]

    if missing_cols:
        logging.warning(f"Target columns missing from fetched Statcast data: {missing_cols}. These columns will be NULL in the DB.")
        # Add missing columns filled with None (or appropriate default)
        for col in missing_cols:
            df[col] = None # Or pd.NA

    df_cleaned = df[cols_to_keep].copy()

    # --- Data Type Conversions (Add more as needed based on schema) ---
    # Convert game_date to string format compatible with DB TIMESTAMP (if needed)
    if 'game_date' in df_cleaned.columns:
         # pybaseball usually returns datetime objects, format if necessary
         # Example: df_cleaned['game_date'] = pd.to_datetime(df_cleaned['game_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
         # Ensure it's datetime before potential conversion
         df_cleaned['game_date'] = pd.to_datetime(df_cleaned['game_date'], errors='coerce')


    # Convert potential float IDs to integers where appropriate (handle NaNs)
    int_cols = ['batter', 'pitcher', 'zone', 'balls', 'strikes', 'game_year', 'outs_when_up',
                 'inning', 'hit_location', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5',
                 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'game_pk', 'pitcher.1',
                 'fielder_2.1', 'at_bat_number', 'pitch_number', 'home_score', 'away_score',
                 'bat_score', 'fld_score', 'post_away_score', 'post_home_score',
                 'post_bat_score', 'post_fld_score', 'spin_axis', 'on_1b', 'on_2b', 'on_3b'] # Add relevant int cols from schema
    for col in int_cols:
        if col in df_cleaned.columns:
            # Convert to nullable integer type (Int64) to handle potential NaNs/None
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')


    # Convert potential object columns to numeric where appropriate
    numeric_cols = ['release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
                    'plate_x', 'plate_z', 'hc_x', 'hc_y', 'sz_top', 'sz_bot', 'hit_distance_sc',
                    'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate',
                    'release_extension', 'release_pos_y', 'estimated_ba_using_speedangle',
                    'estimated_woba_using_speedangle', 'woba_value', 'bat_speed', 'swing_length',
                    'delta_home_win_exp', 'delta_run_exp'] # Add relevant numeric cols from schema
    for col in numeric_cols:
         if col in df_cleaned.columns:
              df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    logging.info(f"Data cleaning complete. {len(df_cleaned)} rows prepared.")
    return df_cleaned


def append_to_db(df, db_path, table_name):
    """Appends the DataFrame to the specified SQLite table."""
    if df.empty:
        logging.info("No data to append to the database.")
        return 0

    logging.info(f"Appending {len(df)} rows to table '{table_name}' in {db_path}...")
    rows_added = 0
    try:
        with sqlite3.connect(db_path) as conn:
            # Use pandas to_sql for easier appending
            df.to_sql(name=table_name, con=conn, if_exists='append', index=False)
            rows_added = len(df)
            logging.info(f"Successfully appended {rows_added} rows.")
    except sqlite3.Error as e:
        logging.error(f"SQLite error appending data to '{table_name}': {e}")
        logging.error("Data that failed to append (first 5 rows):")
        try:
            logging.error(df.head().to_string()) # Log first few rows of failing data
        except Exception as log_e:
             logging.error(f"Could not log failing data: {log_e}")
    except Exception as e:
        logging.error(f"Unexpected error appending data: {e}")
        logging.error(traceback.format_exc())

    return rows_added

# --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info("--- Starting MLB Data Update Script ---")

    # 1. Get target table columns
    target_columns = get_target_columns(DB_PATH, TABLE_NAME)
    if not target_columns:
        logging.error("Could not get target columns. Aborting.")
        sys.exit(1)

    # 2. Define overall date range
    try:
        start_date = datetime.strptime(FETCH_START_DATE, '%Y-%m-%d').date()
        end_date = datetime.strptime(FETCH_END_DATE, '%Y-%m-%d').date()
        logging.info(f"Target date range: {start_date} to {end_date}")
    except ValueError:
        logging.error(f"Invalid date format in FETCH_START_DATE or FETCH_END_DATE. Use YYYY-MM-DD.")
        sys.exit(1)

    # 3. Loop through date range in chunks
    current_date = start_date
    total_rows_added = 0
    while current_date <= end_date:
        chunk_end_date = min(current_date + timedelta(days=CHUNK_DAYS - 1), end_date)
        start_dt_str = current_date.strftime('%Y-%m-%d')
        end_dt_str = chunk_end_date.strftime('%Y-%m-%d')

        # Fetch data for the chunk
        fetched_df = fetch_statcast_data_chunk(start_dt_str, end_dt_str)

        if not fetched_df.empty:
            # Clean and prepare
            cleaned_df = clean_and_prepare_data(fetched_df, target_columns)
            # Append to DB
            rows_added = append_to_db(cleaned_df, DB_PATH, TABLE_NAME)
            total_rows_added += rows_added
            # Optional: Explicitly delete dataframes to free memory
            del fetched_df
            del cleaned_df
            import gc
            gc.collect()

        # Move to the next chunk start date
        current_date = chunk_end_date + timedelta(days=1)

        # Add a delay before the next fetch
        if current_date <= end_date:
            logging.info(f"Waiting {FETCH_DELAY} seconds before next chunk...")
            time.sleep(FETCH_DELAY)

    logging.info(f"--- Data Update Script Finished ---")
    logging.info(f"Total rows added to '{TABLE_NAME}': {total_rows_added}")

