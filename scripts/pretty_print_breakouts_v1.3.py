#!/usr/bin/env python3
# pretty_print_breakouts.py
# Reads the enriched breakout performance data and prints it to the terminal
# with flair, including player names, statlines, context, and sorted by date/score.

import sqlite3
import pandas as pd
import numpy as np
import subprocess
import shutil
import os
import logging
import requests # For API fallback
from functools import lru_cache # For caching API calls
from datetime import datetime # For formatting date output

# --- Configuration ---
DB_PATH = "breakouts.db"
TABLE_NAME = "breakout_performances_topN_enriched" # <<-- Read from the enriched table
MAIN_DB_PATH = "mlb_data2.db" # Path to main database for player names
PEOPLE_TABLE_NAME = "people" # Assumed name for player lookup table

FIGLET_FONT = "miniwi" # User preferred font
TITLE_TEXT = "Daily Breakout Report"
SECTION_SEPARATOR = "=" * 44 # Separator length
MAX_ROWS_DISPLAY = 50 # Limit the total number of rows printed to the terminal

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
    try:
        player_id_int = int(player_id)
    except (ValueError, TypeError):
        logging.warning(f"Invalid player_id format for API call: {player_id}")
        return f"ID:{player_id}"

    url = f"https://statsapi.mlb.com/api/v1/people/{player_id_int}"
    try:
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        if data.get('people') and len(data['people']) > 0:
            # Attempt to get primary position as well
            primary_pos = data['people'][0].get('primaryPosition', {}).get('abbreviation', 'UNK')
            full_name = data['people'][0].get('fullName', f"ID:{player_id}")
            return full_name, primary_pos # Return name and position
        else:
            logging.warning(f"No 'people' data found in API response for ID {player_id}")
            return f"ID:{player_id}", "UNK"
    except requests.exceptions.RequestException as e:
        logging.warning(f"API request failed for player ID {player_id}: {e}")
        return f"ID:{player_id}", "UNK"
    except Exception as e:
        logging.warning(f"Error parsing API response for player ID {player_id}: {e}")
        return f"ID:{player_id}", "UNK"


def load_player_name_map(db_path, table_name):
    """Attempts to load player ID to name map from a local database table."""
    logging.info(f"Attempting to load player names from {db_path}, table {table_name}...")
    if not os.path.exists(db_path):
        logging.warning(f"Main database file not found: {db_path}. Will use API fallback for names.")
        return None, None # Return None for map and positions

    name_map = {}
    pos_map = {} # Add position map
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            if not cursor.fetchone():
                logging.warning(f"Table '{table_name}' not found in {db_path}. Will use API fallback for names.")
                return None, None

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [info[1].lower() for info in cursor.fetchall()]
            id_col_options = ['playerid', 'player_id', 'id']
            first_name_col_options = ['namefirst', 'firstname', 'first_name']
            last_name_col_options = ['namelast', 'lastname', 'last_name']
            # Add options for position column if available in people table
            pos_col_options = ['pos', 'primaryposition', 'primary_position']

            id_col = next((col for col in id_col_options if col in columns), None)
            first_name_col = next((col for col in first_name_col_options if col in columns), None)
            last_name_col = next((col for col in last_name_col_options if col in columns), None)
            pos_col = next((col for col in pos_col_options if col in columns), None) # Find position column

            if not (id_col and first_name_col and last_name_col):
                 logging.warning(f"Expected columns (ID, First Name, Last Name) not found in table '{table_name}'. Will use API fallback.")
                 return None, None

            # Fetch data including position if available
            select_cols = [f'"{id_col}"', f'"{first_name_col}"', f'"{last_name_col}"']
            if pos_col:
                 select_cols.append(f'"{pos_col}"')
            query = f"SELECT {', '.join(select_cols)} FROM {table_name}"

            logging.debug(f"Executing name query: {query}")
            df_names = pd.read_sql_query(query, conn)

            df_names['fullName'] = df_names[first_name_col].fillna('') + ' ' + df_names[last_name_col].fillna('')
            df_names['fullName'] = df_names['fullName'].str.strip()
            df_names[id_col] = pd.to_numeric(df_names[id_col], errors='coerce')
            df_names.dropna(subset=[id_col], inplace=True)
            try:
                df_names[id_col] = df_names[id_col].astype(int)
                name_map = pd.Series(df_names['fullName'].values, index=df_names[id_col]).to_dict()
                # Create position map if column was found
                if pos_col:
                     pos_map = pd.Series(df_names[pos_col].fillna('UNK').values, index=df_names[id_col]).to_dict()
                     logging.info(f"Successfully loaded {len(name_map)} player names and {len(pos_map)} positions from local DB.")
                else:
                     logging.info(f"Successfully loaded {len(name_map)} player names from local DB (position not found).")
                return name_map, pos_map
            except ValueError as ve:
                 logging.error(f"Could not convert player IDs in '{table_name}' to integers: {ve}. Using API fallback.")
                 return None, None
    except sqlite3.Error as e:
        logging.error(f"SQLite error loading player names: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading player names: {e}")
        return None, None

# --- Main Logic ---

def load_breakout_data(db_path, table_name):
    """Loads breakout data from the SQLite database."""
    logging.info(f"Loading data from {db_path}, table {table_name}...")
    if not os.path.exists(db_path):
        logging.error(f"Database file not found: {db_path}")
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            if not cursor.fetchone():
                logging.error(f"Table '{table_name}' not found in the database.")
                return None
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            logging.info(f"Successfully loaded {len(df)} rows.")

            if 'player_id' not in df.columns: return None
            df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce').astype('Int64')
            df.dropna(subset=['player_id'], inplace=True)
            df['player_id'] = df['player_id'].astype(int)

            if 'game_date' not in df.columns: return None
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
            df.dropna(subset=['game_date'], inplace=True)

            # Convert stat columns to numeric, filling errors/NaNs with 0
            stat_cols = ['IP', 'H', 'R', 'ER', 'BB', 'SO', 'HR', 'outs_recorded',
                         'single', 'double', 'triple', 'AB', 'SB', 'CS', 'RBI', 'PA', 'daily_score']
            for col in stat_cols:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                 else:
                      logging.warning(f"Stat column '{col}' not found in loaded data. Will default to 0.")
                      df[col] = 0

            # Convert score columns to numeric, keeping NaNs for placeholders
            score_cols = ['actual_home_score', 'actual_away_score']
            for col in score_cols:
                 if col in df.columns:
                      df[col] = pd.to_numeric(df[col], errors='coerce') # Keep NaNs
                 else:
                      df[col] = np.nan

            return df
    except sqlite3.Error as e:
        logging.error(f"SQLite error loading data: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def add_player_names_and_pos(df, name_map, pos_map):
    """Adds player names and positions using maps or API fallback."""
    if 'player_id' not in df.columns:
        logging.error("Missing 'player_id' column in DataFrame. Cannot add names/pos.")
        df['player_name'] = 'Error: Missing ID'
        df['position'] = 'UNK'
        return df

    df['player_id'] = df['player_id'].astype(int) # Ensure correct type

    # --- Get Names ---
    if name_map:
        logging.info("Mapping names using local lookup...")
        df['player_name'] = df['player_id'].map(name_map)
        missing_names = df['player_name'].isnull()
        if missing_names.any():
            logging.warning(f"{missing_names.sum()} player names not found locally. Using API fallback...")
            # Apply API only for missing names
            api_results = df.loc[missing_names, 'player_id'].apply(lambda x: get_player_name_api(int(x)))
            df.loc[missing_names, 'player_name'] = api_results.apply(lambda x: x[0]) # Get name from tuple
            # If position map doesn't exist, try to get position from API result too
            if not pos_map:
                 if 'position' not in df.columns: df['position'] = 'UNK' # Initialize if needed
                 df.loc[missing_names, 'position'] = api_results.apply(lambda x: x[1]) # Get pos from tuple
        df['player_name'].fillna(df['player_id'].apply(lambda x: f"ID:{x}"), inplace=True)
    else:
        logging.info("No local name map. Using API fallback for all names and positions...")
        api_results = df['player_id'].apply(lambda x: get_player_name_api(int(x)))
        df['player_name'] = api_results.apply(lambda x: x[0])
        df['position'] = api_results.apply(lambda x: x[1])

    # --- Get Positions (if not already fetched from API) ---
    if pos_map:
         logging.info("Mapping positions using local lookup...")
         df['position'] = df['player_id'].map(pos_map).fillna('UNK')
    elif 'position' not in df.columns: # Ensure column exists if neither map nor API provided it
         df['position'] = 'UNK'

    return df


def format_statline(row):
    """ Creates the formatted statline string based on player type. """
    is_pitcher = row.get('IP', 0) > 0
    position = row.get('position', 'UNK') # Get position if available

    if is_pitcher:
        ip_full = int(row.get('IP', 0))
        ip_partial = int(round((row.get('IP', 0) - ip_full) * 3))
        ip_str = f"{ip_full}.{ip_partial}" if ip_partial > 0 else f"{ip_full}.0"
        stats = [
            f"Position: {position}",
            f"IP: {ip_str}",
            f"H: {int(row.get('H', 0))}", # Placeholder value
            f"R: {int(row.get('R', 0))}", # Placeholder value
            f"ER: {int(row.get('ER', 0))}", # Placeholder value
            f"BB: {int(row.get('BB', 0))}",
            f"SO: {int(row.get('SO', 0))}"
        ]
    else: # Hitter
        h_calc = int(row.get('single', 0) + row.get('double', 0) + row.get('triple', 0) + row.get('HR', 0))
        h_val = int(row.get('H', h_calc))
        stats = [
            f"Position: {position}",
            f"AB: {int(row.get('AB', 0))}", # Placeholder value
            f"R: {int(row.get('R', 0))}", # Placeholder value
            f"H: {h_val}",
            f"2B: {int(row.get('double', 0))}",
            f"3B: {int(row.get('triple', 0))}",
            f"HR: {int(row.get('HR', 0))}",
            f"RBI: {int(row.get('RBI', 0))}", # Placeholder value
            f"BB: {int(row.get('BB', 0))}",
            f"SO: {int(row.get('SO', 0))}",
            f"SB: {int(row.get('SB', 0))}" # Placeholder value
        ]
    return "\n".join(stats)

def format_team_result(row):
    """ Formats the team result string using actual scores if available. """
    home_score = row.get('actual_home_score')
    away_score = row.get('actual_away_score')
    winner = row.get('game_winner', '?') # Placeholder if winner isn't stored

    # Try to determine winner if scores are present but winner isn't
    if winner == '?' and pd.notna(home_score) and pd.notna(away_score):
        if home_score > away_score: winner = 'H' # Assuming 'H' for home win
        elif away_score > home_score: winner = 'A' # Assuming 'A' for away win
        else: winner = 'T' # Tie

    # Format scores, handling potential NaNs
    home_score_str = str(int(home_score)) if pd.notna(home_score) else '?'
    away_score_str = str(int(away_score)) if pd.notna(away_score) else '?'

    # Determine W/L based on winner (needs player's team context - placeholder for now)
    outcome = '(W/L/?)' # Default placeholder

    # Get team names (assuming they were merged during enrichment)
    home_team = row.get('home_team', 'Home')
    away_team = row.get('away_team', 'Away')

    return f"TEAM RESULT: {outcome} {away_team} {away_score_str} - {home_team} {home_score_str}"


def display_data(df):
    """Formats and prints the data to the terminal in the desired format."""
    if df is None or df.empty:
        logging.warning("No breakout data to display.")
        return

    has_figlet = is_tool("figlet")
    has_lolcat = is_tool("lolcat")

    # --- Print Title ---
    title_output = TITLE_TEXT
    if has_figlet:
        figlet_command = ["figlet", "-f", FIGLET_FONT, TITLE_TEXT]
        generated_title = run_command(figlet_command)
        if generated_title: title_output = generated_title
        else: logging.warning(f"figlet font '{FIGLET_FONT}' not found or figlet failed.")

    if has_lolcat:
        lolcat_title = run_command(["lolcat"], input_text=title_output)
        print(lolcat_title if lolcat_title else title_output)
    else:
        print(title_output)

    # --- Prepare Data ---
    display_df = df.copy()

    # Sort by date (ascending), then score (descending)
    if 'game_date' in display_df.columns and 'daily_score' in display_df.columns:
        logging.info("Sorting data by game_date (ascending) and daily_score (descending)...")
        display_df['game_date'] = pd.to_datetime(display_df['game_date'])
        display_df = display_df.sort_values(by=['game_date', 'daily_score'], ascending=[True, False])
    else:
         logging.warning("Could not sort data properly due to missing 'game_date' or 'daily_score'.")

    # Limit rows AFTER sorting
    display_df = display_df.head(MAX_ROWS_DISPLAY)

    # --- Print Breakouts ---
    current_date_str = None
    for index, row in display_df.iterrows():
        row_date = row['game_date']
        # Format date with ordinal suffix (e.g., 1st, 2nd, 3rd, 4th)
        day = row_date.day
        if 11 <= day <= 13: suffix = 'th'
        else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        row_date_str = row_date.strftime(f'%B {day}{suffix}, %Y') # e.g., May 5th, 2025

        # Print Date Header if it's a new date
        if row_date_str != current_date_str:
            print("\n" + SECTION_SEPARATOR)
            date_header = f"{row_date_str}"
            if has_lolcat:
                 lolcat_header = run_command(["lolcat", "-a", "-d", "1", "-p", "0.8"], input_text=date_header)
                 print(lolcat_header if lolcat_header else date_header)
            else:
                 print(date_header)
            print(SECTION_SEPARATOR)
            current_date_str = row_date_str

        # Format Statline and Team Result
        statline = format_statline(row)
        team_result = format_team_result(row) # Format the team result string

        # Format Player Info
        player_info = [
            f"\nPlayer: {row['player_name']} ({row['player_id']})",
            f"Predicted Score: {row['daily_score']:.2f}", # Display score
            statline,
            team_result # Use the formatted team result
        ]
        player_output = "\n".join(player_info)

        # Print Player Block (optionally with lolcat)
        if has_lolcat:
            lolcat_player = run_command(["lolcat", "-p", "0.5", "-S", "10"], input_text=player_output)
            print(lolcat_player if lolcat_player else player_output)
        else:
            print(player_output)

        print("-----") # Separator between players


if __name__ == "__main__":
    # 1. Load breakout data
    breakout_data = load_breakout_data(DB_PATH, TABLE_NAME)

    if breakout_data is not None and not breakout_data.empty:
        # 2. Attempt to load local player name and position map
        player_names, player_positions = load_player_name_map(MAIN_DB_PATH, PEOPLE_TABLE_NAME)

        # 3. Add names and positions to the breakout data
        breakout_data_with_context = add_player_names_and_pos(breakout_data, player_names, player_positions)

        # 4. Display the data
        display_data(breakout_data_with_context)
    else:
        logging.info("No breakout data found to display.")
