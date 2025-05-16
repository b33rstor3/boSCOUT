#!/usr/bin/env python3
# pretty_print_breakouts.py
# Reads the breakout performance data and prints it to the terminal with flair,
# including player names.

import sqlite3
import pandas as pd
import numpy as np # <--- Added numpy import
import subprocess
import shutil
import os
import logging
import requests # Added for API fallback
from functools import lru_cache # For caching API calls

# --- Configuration ---
DB_PATH = "breakouts.db"
TABLE_NAME = "breakout_performances"
MAIN_DB_PATH = "mlb_data2.db" # Path to main database for player names
PEOPLE_TABLE_NAME = "people" # Assumed name for player lookup table

FIGLET_FONT = "miniwi" # User preferred font
TITLE_TEXT = "Breakout Performances"
# Columns to display in the table
COLUMNS_TO_DISPLAY = [
    'game_date',
    'player_name', # Changed from player_id
    'daily_score',
    'roll_30d_score_percentile',
    'roll_30d_daily_score_mean',
    'roll_30d_daily_score_median_threshold'
    # Add enriched columns here later, e.g., 'ballpark', 'opponent_team'
]
MAX_ROWS_DISPLAY = 50 # Limit the number of rows printed to the terminal

# --- Logging Setup ---
# Remove existing handlers to avoid duplicate logs if script is re-run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Functions ---

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def run_command(command_list, input_text=None):
    """Runs a shell command and returns its output."""
    try:
        process = subprocess.run(
            command_list,
            input=input_text,
            text=True,
            capture_output=True,
            check=True # Raise exception on non-zero exit code
        )
        return process.stdout
    except FileNotFoundError:
        # Don't log error here, handled by caller checking None return
        # logging.error(f"Error: Command '{command_list[0]}' not found.")
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command '{' '.join(command_list)}': {e}")
        logging.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred running command '{' '.join(command_list)}': {e}")
        return None

@lru_cache(maxsize=None) # Cache API calls to avoid redundant lookups
def get_player_name_api(player_id):
    """Fetches player name from MLB Stats API as a fallback."""
    # Convert player_id to int if it's not already, handle potential errors
    try:
        player_id_int = int(player_id)
    except (ValueError, TypeError):
        logging.warning(f"Invalid player_id format for API call: {player_id}")
        return f"ID:{player_id}" # Return original ID if conversion fails

    url = f"https://statsapi.mlb.com/api/v1/people/{player_id_int}"
    try:
        res = requests.get(url, timeout=5) # Add timeout
        res.raise_for_status() # Check for HTTP errors
        data = res.json()
        if data.get('people') and len(data['people']) > 0:
            return data['people'][0].get('fullName', f"ID:{player_id}")
        else:
            logging.warning(f"No 'people' data found in API response for ID {player_id}")
            return f"ID:{player_id}"
    except requests.exceptions.RequestException as e:
        logging.warning(f"API request failed for player ID {player_id}: {e}")
        return f"ID:{player_id}" # Return ID on API error
    except Exception as e:
        logging.warning(f"Error parsing API response for player ID {player_id}: {e}")
        return f"ID:{player_id}" # Return ID on other errors


def load_player_name_map(db_path, table_name):
    """Attempts to load player ID to name map from a local database table."""
    logging.info(f"Attempting to load player names from {db_path}, table {table_name}...")
    if not os.path.exists(db_path):
        logging.warning(f"Main database file not found: {db_path}. Will use API fallback for names.")
        return None

    name_map = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            if not cursor.fetchone():
                logging.warning(f"Table '{table_name}' not found in {db_path}. Will use API fallback for names.")
                return None

            # Check for expected columns (adjust column names if necessary)
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [info[1].lower() for info in cursor.fetchall()] # Use lower case for matching
            # Try common variations for player ID and name columns
            id_col_options = ['playerid', 'player_id', 'id']
            first_name_col_options = ['namefirst', 'firstname', 'first_name']
            last_name_col_options = ['namelast', 'lastname', 'last_name']

            id_col = next((col for col in id_col_options if col in columns), None)
            first_name_col = next((col for col in first_name_col_options if col in columns), None)
            last_name_col = next((col for col in last_name_col_options if col in columns), None)


            if not (id_col and first_name_col and last_name_col):
                 logging.warning(f"Expected columns (ID, First Name, Last Name) not found in table '{table_name}'. Will use API fallback.")
                 return None

            # Fetch data using the found column names
            query = f'SELECT "{id_col}", "{first_name_col}", "{last_name_col}" FROM {table_name}'
            logging.debug(f"Executing name query: {query}")
            df_names = pd.read_sql_query(query, conn)

            # Create map, handling potential nulls
            df_names['fullName'] = df_names[first_name_col].fillna('') + ' ' + df_names[last_name_col].fillna('')
            df_names['fullName'] = df_names['fullName'].str.strip()
            # Convert playerID to appropriate type (int) for matching
            df_names[id_col] = pd.to_numeric(df_names[id_col], errors='coerce')
            df_names.dropna(subset=[id_col], inplace=True)
            # Ensure ID column is integer before setting index
            try:
                df_names[id_col] = df_names[id_col].astype(int)
                name_map = pd.Series(df_names['fullName'].values, index=df_names[id_col]).to_dict()
                logging.info(f"Successfully loaded {len(name_map)} player names from local DB.")
                return name_map
            except ValueError as ve:
                 logging.error(f"Could not convert player IDs in '{table_name}' to integers: {ve}. Using API fallback.")
                 return None


    except sqlite3.Error as e:
        logging.error(f"SQLite error loading player names: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading player names: {e}")
        return None


# --- Main Logic ---

def load_breakout_data(db_path, table_name):
    """Loads breakout data from the SQLite database."""
    logging.info(f"Loading data from {db_path}, table {table_name}...")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            if not cursor.fetchone():
                logging.error(f"Table '{table_name}' not found in the database.")
                return None
            # Read data
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            logging.info(f"Successfully loaded {len(df)} rows.")
            # Ensure player_id is integer
            if 'player_id' in df.columns:
                 df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').astype('Int64')
                 df.dropna(subset=['player_id'], inplace=True) # Drop rows where ID is bad
                 df['player_id'] = df['player_id'].astype(int)
            else:
                 logging.error("Column 'player_id' not found in breakout data.")
                 return None
            return df
    except sqlite3.Error as e:
        logging.error(f"SQLite error loading data: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def add_player_names(df, name_map):
    """Adds player names to the DataFrame using the map or API fallback."""
    if 'player_id' not in df.columns:
        logging.error("Missing 'player_id' column in DataFrame. Cannot add names.")
        df['player_name'] = 'Error: Missing ID'
        return df

    # Ensure player_id is the correct type for mapping
    df['player_id'] = df['player_id'].astype(int)

    if name_map:
        logging.info("Mapping names using local lookup...")
        df['player_name'] = df['player_id'].map(name_map)
        # Handle names not found in the local map using API fallback
        missing_names_count = df['player_name'].isnull().sum()
        if missing_names_count > 0:
            logging.warning(f"{missing_names_count} player names not found in local map. Using API fallback...")
            # Apply API call only to rows where name is missing
            # Make sure to pass integer IDs to the API function
            df.loc[df['player_name'].isnull(), 'player_name'] = df.loc[df['player_name'].isnull(), 'player_id'].apply(lambda x: get_player_name_api(int(x)))
        # Fill any remaining NaNs (e.g., if API failed)
        df['player_name'].fillna(df['player_id'].apply(lambda x: f"ID:{x}"), inplace=True)
    else:
        logging.info("No local name map found. Using API fallback for all names...")
        df['player_name'] = df['player_id'].apply(lambda x: get_player_name_api(int(x)))

    return df


def display_data(df):
    """Formats and prints the data to the terminal."""
    if df is None or df.empty:
        logging.warning("No breakout data to display.")
        return

    # Check for required tools
    has_figlet = is_tool("figlet")
    has_lolcat = is_tool("lolcat")

    # --- Print Title ---
    title_output = TITLE_TEXT
    if has_figlet:
        # Check if the specific font exists for figlet
        figlet_command = ["figlet", "-f", FIGLET_FONT, TITLE_TEXT]
        generated_title = run_command(figlet_command)
        if generated_title:
            title_output = generated_title
        else:
            logging.warning(f"figlet font '{FIGLET_FONT}' might not be available. Using default figlet.")
            generated_title_default = run_command(["figlet", TITLE_TEXT])
            if generated_title_default:
                 title_output = generated_title_default
            else:
                 logging.warning("figlet command failed even with default font.")
                 # Fallback to plain text handled by title_output initial value

    if has_lolcat:
        lolcat_title = run_command(["lolcat"], input_text=title_output)
        if lolcat_title:
            print(lolcat_title)
        else:
            print(title_output) # Print plain title if lolcat fails
    else:
        print(title_output) # Print plain title if lolcat not available

    print("\n" + "-" * 80) # Separator

    # --- Prepare Data for Table ---
    display_df = df.copy()

    # Filter columns to only those that exist in the DataFrame and are requested
    existing_display_cols = [col for col in COLUMNS_TO_DISPLAY if col in display_df.columns]
    if not existing_display_cols:
         logging.error("None of the specified COLUMNS_TO_DISPLAY exist in the loaded data.")
         return

    display_df = display_df[existing_display_cols]

    # Sort by score (descending) and limit rows
    if 'daily_score' in display_df.columns:
        display_df = display_df.sort_values(by='daily_score', ascending=False)
    display_df = display_df.head(MAX_ROWS_DISPLAY)

    # Format specific columns (e.g., floats to 2 decimal places)
    for col in display_df.select_dtypes(include=['float', 'float64']).columns:
        # Check if column name suggests percentile before formatting
        if 'percentile' in col.lower():
             # Handle potential errors during formatting
             try:
                 # Ensure the column is numeric before formatting
                 numeric_col = pd.to_numeric(display_df[col], errors='coerce')
                 display_df[col] = numeric_col.map('{:,.1f}%'.format, na_action='ignore')
             except (TypeError, ValueError) as e:
                  logging.warning(f"Could not format column '{col}' as percentile: {e}")
                  display_df[col] = display_df[col].astype(str) # Convert to string as fallback
        else:
             try:
                 # Ensure the column is numeric before formatting
                 numeric_col = pd.to_numeric(display_df[col], errors='coerce')
                 display_df[col] = numeric_col.map('{:,.2f}'.format, na_action='ignore')
             except (TypeError, ValueError) as e:
                  logging.warning(f"Could not format column '{col}' as float: {e}")
                  display_df[col] = display_df[col].astype(str) # Convert to string as fallback

    # Replace any remaining None/NaN/NaT with empty strings for display
    display_df.fillna('', inplace=True)

    # --- Print Table ---
    try:
        from tabulate import tabulate
        # Adjust column alignment if needed (e.g., left-align names)
        colalign = ["left"] * len(display_df.columns)
        # Find index of numeric-like columns for right alignment
        for i, col in enumerate(display_df.columns):
            # Check if original dtype was number OR if name suggests numeric content
             is_numeric_like = pd.api.types.is_numeric_dtype(df[col]) or \
                               any(k in col.lower() for k in ['score', 'mean', 'median', 'percentile'])
             if is_numeric_like and col != 'player_id': # Don't right-align ID
                  colalign[i] = "right"
             if col == 'player_name': # Ensure name is left-aligned
                  colalign[i] = "left"


        table_output = tabulate(
            display_df,
            headers='keys',
            tablefmt='fancy_grid', # 'psql' or 'simple' are also good options
            showindex=False,
            colalign=colalign
        )
    except ImportError:
        logging.warning("`tabulate` library not found. Printing basic DataFrame string.")
        logging.warning("Install it using: pip install tabulate")
        table_output = display_df.to_string(index=False)
    except Exception as e:
         logging.error(f"Error creating table with tabulate: {e}")
         table_output = display_df.to_string(index=False) # Fallback

    if has_lolcat:
        # Pipe table output to lolcat
        lolcat_table = run_command(["lolcat", "-p", "1.0", "-S", "0"], input_text=table_output) # Adjusted lolcat options
        if lolcat_table:
            print(lolcat_table)
        else:
            logging.warning("lolcat command failed to process table output.")
            print(table_output) # Print plain table if lolcat fails
    else:
        print(table_output) # Print plain table if lolcat not available


if __name__ == "__main__":
    # 1. Load breakout data
    breakout_data = load_breakout_data(DB_PATH, TABLE_NAME)

    if breakout_data is not None and not breakout_data.empty:
        # 2. Attempt to load local player name map
        player_names = load_player_name_map(MAIN_DB_PATH, PEOPLE_TABLE_NAME)

        # 3. Add names to the breakout data
        breakout_data_with_names = add_player_names(breakout_data, player_names)

        # 4. Display the data
        display_data(breakout_data_with_names)
    else:
        logging.info("No breakout data found to display.")

