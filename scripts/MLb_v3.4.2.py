
# ==============================================================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, date, timedelta
import sqlite3
import requests
from tqdm import tqdm
import gc
import sys
import traceback
import json
import warnings
import time
import os
import logging
import pickle
import meteostat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import meteostat and provide clear instructions if missing
try:
    from meteostat import Point, Daily
except ImportError:
    logging.error("ERROR: meteostat library not found. Please install it: pip install meteostat")
    # IMPROVEMENT: Exit only if historical weather is deemed critical for training.
    # sys.exit(1) # Exit if meteostat is essential for core functionality (historical weather)
    logging.warning("meteostat library not found. Historical weather features will use defaults.")
    Point, Daily = None, None # Set to None to handle gracefully later

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================
# CONSTANTS & CONFIG
# ==============================================================
# --- File Paths ---
BASE_FILENAME = "mlb_model_v2_scored_weather"
BATTER_MODEL_PATH = f"{BASE_FILENAME}_batter.pkl"
PITCHER_MODEL_PATH = f"{BASE_FILENAME}_pitcher.pkl"
BATTER_FEATURES_PATH = f"{BASE_FILENAME}_batter_features.pkl"
PITCHER_FEATURES_PATH = f"{BASE_FILENAME}_pitcher_features.pkl"
EVALUATION_REPORT_PATH = f"{BASE_FILENAME}_evaluation_report.json" # NOTE: Evaluation report generation not implemented in this script version
DB_PATH = "mlb_data2.db" # Assumed DB name

# --- Model & Feature Parameters ---
N_GAMES = 15 # Window size for rolling features
TOP_N = 20 # Number of top performers to list
CHUNKSIZE = 5000 # Rows to process per chunk from DB (Adjust based on memory)
HISTORICAL_DAYS_FOR_FEATURES = 45 # Number of past days to load for feature generation during prediction
# FIX [I-06]: Define number of splits for TimeSeriesSplit
N_SPLITS_TIME_SERIES = 5

# --- League Averages & Scoring Constants ---
# NOTE [I-02]: These should ideally be calculated dynamically from the training dataset's time period
# or sourced from reliable year-specific baseball statistics providers. Using static placeholders.
LEAGUE_AVERAGES = {
    # Basic Stats (Placeholders/Typical)
    'release_speed': 93.0,
    'spin_rate': 2400.0, # Renamed from release_spin_rate for consistency if needed elsewhere
    'launch_speed': 88.0,
    'launch_angle': 12.0,
    'estimated_ba_using_speedangle': 0.245,
    'estimated_woba_using_speedangle': 0.315, # Kept for potential feature use
    'pfx_x': 0.0,
    'pfx_z': 0.5,
    # Weather Defaults (Typical)
    'temp': 70.0, # degrees F
    'humidity': 50.0, # percentage
    'wind_speed': 5.0, # mph
    'wind_dir': 0.0, # degrees (0=North, 90=East, etc.) - Check API source convention
    'pressure': 1013.25, # hPa (millibars)
    # --- Constants required by NEW scoring formulas ---
    'league_woba': 0.315, # League average wOBA (Weighted On-Base Average) [1, 2]
    'wOBA_scale': 1.2, # Typical wOBA scale factor (approx 1.15-1.25) [1, 3, 4]
    'FIP_constant': 3.10, # Fielding Independent Pitching constant (adjust based on league year) [5, 6, 7, 8]
    #!!! IMPORTANT: lgFIP should be calculated dynamically from your data!!! [6, 7]
    'lgFIP': 3.10, # Placeholder for League Average FIP
    'runs_per_win': 9.0 # Estimated runs needed per win (can range 9-10) [9, 10, 11]
}
# NOTE: Barrel rate constant removed as barrel is calculated dynamically if needed

# --- APIs ---
# MLB Stats API Endpoint for Schedules
MLB_SCHEDULE_API_URL = "https://statsapi.mlb.com/api/v1/schedule"
# Weather API
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
# IMPROVEMENT: Use environment variable for API key security [12, 13]
WEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY', 'bd764f44aa2394a64b4baa82c173787f') # Replace placeholder or set env var

# --- Ballpark Coordinates (Add more as needed) ---
# Source: Simple search or baseball reference sites
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
    'NYY': {'lat': 40.8296, 'lon': -73.9262}, 'OAK': {'lat': 37.7510, 'lon': -122.2009}, # Check current status
    'PHI': {'lat': 39.9061, 'lon': -75.1665}, 'PIT': {'lat': 40.4469, 'lon': -80.0057},
    'SD': {'lat': 32.7076, 'lon': -117.1570}, 'SEA': {'lat': 47.5914, 'lon': -122.3325},
    'SF': {'lat': 37.7786, 'lon': -122.3893}, 'STL': {'lat': 38.6226, 'lon': -90.1928},
    'TB': {'lat': 27.7682, 'lon': -82.6534}, 'TEX': {'lat': 32.7513, 'lon': -97.0828},
    'TOR': {'lat': 43.6414, 'lon': -79.3894}, 'WSH': {'lat': 38.8730, 'lon': -77.0074},
    # Default coordinates (center of US) if team not found
    'DEFAULT': {'lat': 39.8283, 'lon': -98.5795}
}
# NOTE [I-15]: Map MLB API team abbreviations (if different) to Ballpark Coords keys if needed
# This map should be verified against the actual abbreviations returned by the MLB Stats API schedule endpoint.
TEAM_ABBR_MAP = {
    'CWS': 'CWS', 'CHW': 'CWS', # Example: Map API's 'CHW' to our 'CWS'
    'CHC': 'CHC',
    'NYM': 'NYM',
    'NYY': 'NYY',
    'LAA': 'LAA', 'ANA': 'LAA', # Example mapping
    'SD': 'SD', 'SDP': 'SD',   # Example mapping
    'SF': 'SF', 'SFG': 'SF',   # Example mapping
    'TB': 'TB', 'TBR': 'TB',   # Example mapping
    'KC': 'KC', 'KCR': 'KC',   # Example mapping
    'WSH': 'WSH', 'WSN': 'WSH', # Example mapping
    # Add other mappings as necessary based on schedule API output
}

# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def safe_get_numeric(series, default=0.0):
    """Convert series to numeric, coercing errors and filling NaNs with a default."""
    # IMPROVEMENT: Added explicit check for empty series to avoid potential warnings/errors
    if series is None or series.empty:
        return series # Return empty series if input is empty
    return pd.to_numeric(series, errors='coerce').fillna(default)

def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # FIX [I-14]: Explicitly handle Infinity for JSON compatibility
        if np.isnan(obj):
            return None # Represent NaN as null in JSON
        elif np.isinf(obj):
            # Represent Infinity as a large number string or null, depending on requirements
            # Using None here for simplicity, adjust if a specific string representation is needed
            logging.warning(f"Encountered Infinity value during JSON conversion: {obj}. Replacing with None.")
            return None
        else:
            return float(obj)
    elif isinstance(obj, np.ndarray):
        # IMPROVEMENT: Convert list elements recursively
        return [convert_numpy_types(i) for i in obj.tolist()] # Use tolist() for better handling
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(i) for i in obj]
    # Handle potential non-serializable types gracefully
    try:
        json.dumps(obj) # Test if serializable
        return obj
    except (TypeError, OverflowError):
        logging.debug(f"Could not serialize type {type(obj)}, converting to string.")
        return str(obj) # Convert problematic types to string as a fallback

# ==============================================================
# API & DATA FETCHING FUNCTIONS (Includes NEW Schedule Fetcher)
# ==============================================================

def _make_request_with_retries(url, params=None, headers=None, max_retries=3, timeout=15):
    """Helper function to make HTTP requests with exponential backoff."""
    # IMPROVEMENT [I-10, I-11]: Centralized retry logic for API calls [14]
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            return response
        except requests.exceptions.Timeout:
            logging.warning(f"Attempt {attempt+1}/{max_retries}: Request timed out for {url}.")
        except requests.exceptions.HTTPError as e:
            logging.warning(f"Attempt {attempt+1}/{max_retries}: HTTP Error for {url}: {e}")
            # Check response object existence before accessing status_code
            status_code = response.status_code if response is not None else None
            if status_code == 401:
                logging.error(f"Request failed for {url}: Invalid API key or credentials (401).")
                return None # Don't retry on auth failure
            if status_code == 404:
                logging.error(f"Request failed for {url}: Resource not found (404).")
                return None # Don't retry on not found
            if status_code == 429:
                logging.warning(f"Request failed for {url}: Rate limit likely exceeded (429).")
                # Exponential backoff for rate limiting [14]
                sleep_time = (2 ** attempt) + np.random.uniform(0, 1) # Add jitter
                logging.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                continue # Retry immediately after backoff
            # For other HTTP errors, proceed to standard backoff/retry
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1}/{max_retries}: Request failed for {url} - {e}")
        except Exception as e:
            logging.error(f"Attempt {attempt+1}/{max_retries}: Unexpected error during request for {url}: {e}")

        # Wait before retrying (exponential backoff with jitter)
        if attempt < max_retries - 1:
            sleep_time = (2 ** attempt) + np.random.uniform(0, 1) # Add jitter
            logging.info(f"Retrying request for {url} in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    logging.error(f"All retries failed for request to {url}.")
    return None

def get_schedule_for_date(prediction_date):
    """Fetches MLB schedule for a given date using the MLB Stats API with retries."""
    formatted_date = prediction_date.strftime('%Y-%m-%d')
    # Hydrate parameter includes probablePitcher, team, linescore, flags, venue data [15, 16]
    # NOTE: Getting full lineups often requires different hydration or endpoints, closer to game time.
    params = {
        'sportId': 1,
        'startDate': formatted_date,
        'endDate': formatted_date,
        'hydrate': 'probablePitcher,team,linescore(matchup,runners),flags,venue' # Added linescore for status check
    }
    logging.info(f"Fetching MLB schedule for {formatted_date} from {MLB_SCHEDULE_API_URL}")
    # FIX: Initialize scheduled_games as an empty list
    scheduled_games = []
    all_player_ids = set() # Keep track of players involved (currently just pitchers)

    response = _make_request_with_retries(MLB_SCHEDULE_API_URL, params=params)

    if response is None:
        logging.error(f"Failed to fetch MLB schedule for {formatted_date} after multiple retries.")
        return scheduled_games, all_player_ids # Return empty list and set

    try:
        data = response.json()

        if 'dates' not in data or not data['dates']:
            logging.warning(f"No schedule data found in API response for {formatted_date}.")
            return scheduled_games, all_player_ids # Return empty list and set

        # IMPROVEMENT: Use.get() with default empty list for games
        games = data['dates'][0].get('games', [])
        if not games:
            logging.warning(f"No games listed for {formatted_date}.")
            return scheduled_games, all_player_ids # Return empty list and set

        for game in games:
            # IMPROVEMENT: More robust status checking
            game_status = game.get('status', {}).get('abstractGameState', 'Unknown')
            detailed_status = game.get('status', {}).get('detailedState', 'Unknown')

            # Skip games that are clearly over or postponed
            if game_status in ['Final', 'Game Over', 'Completed Early'] or 'Postponed' in detailed_status:
                logging.debug(f"Skipping game {game.get('gamePk')} with status: {game_status} / {detailed_status}")
                continue

            # Extract necessary info safely using.get()
            home_team_data = game.get('teams', {}).get('home', {}).get('team', {})
            away_team_data = game.get('teams', {}).get('away', {}).get('team', {})
            venue_data = game.get('venue', {})

            home_probable_pitcher_id = game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('id')
            away_probable_pitcher_id = game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('id')

            home_team_id = home_team_data.get('id')
            away_team_id = away_team_data.get('id')
            home_team_abbr = home_team_data.get('abbreviation')
            away_team_abbr = away_team_data.get('abbreviation')
            venue_id = venue_data.get('id')
            venue_name = venue_data.get('name')

            # Map abbreviations using the defined map for consistency with BALLPARK_COORDS
            home_team_abbr_mapped = TEAM_ABBR_MAP.get(home_team_abbr, home_team_abbr)
            away_team_abbr_mapped = TEAM_ABBR_MAP.get(away_team_abbr, away_team_abbr)

            game_info = {
                'game_pk': game.get('gamePk'),
                'game_date': prediction_date, # Use the target date
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_abbr': home_team_abbr_mapped, # Use mapped abbreviation
                'away_team_abbr': away_team_abbr_mapped, # Use mapped abbreviation
                'home_pitcher': home_probable_pitcher_id,
                'away_pitcher': away_probable_pitcher_id,
                'venue_id': venue_id,
                'venue_name': venue_name,
                'status': game_status,
                'detailed_status': detailed_status
            }
            scheduled_games.append(game_info)

            # Add pitcher IDs to the set (handle potential None)
            if home_probable_pitcher_id: all_player_ids.add(home_probable_pitcher_id)
            if away_probable_pitcher_id: all_player_ids.add(away_probable_pitcher_id)

            # NOTE [I-07]: Batter IDs are NOT reliably available from this standard schedule endpoint/hydration.
            # Acquiring batter IDs requires either:
            # 1. A different API endpoint/hydration parameter (if available and documented).
            # 2. Querying team roster endpoints closer to game time.[17, 18]
            # 3. The current workaround: loading all recent batter history in load_and_process_data.

        logging.info(f"Found {len(scheduled_games)} scheduled (non-final/non-postponed) games for {formatted_date}.")
        return scheduled_games, all_player_ids

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding MLB schedule JSON for {formatted_date}: {e}")
        return scheduled_games, all_player_ids # Return empty list and set
    except Exception as e:
        logging.error(f"Unexpected error processing schedule for {formatted_date}: {e}")
        traceback.print_exc()
        return scheduled_games, all_player_ids # Return empty list and set

def get_openweathermap_weather(lat, lon):
    """Fetch current weather data from OpenWeatherMap with retries and error handling."""
    default_weather = {
        'temp': LEAGUE_AVERAGES['temp'], 'humidity': LEAGUE_AVERAGES['humidity'],
        'wind_speed': LEAGUE_AVERAGES['wind_speed'], 'wind_dir': LEAGUE_AVERAGES['wind_dir'],
        'pressure': LEAGUE_AVERAGES['pressure']
    }
    if not WEATHER_API_KEY or WEATHER_API_KEY == 'YOUR_DEFAULT_API_KEY_HERE': # Check if still default placeholder [12]
        logging.warning("Cannot fetch current weather: OpenWeatherMap API key not configured or invalid. Using defaults.")
        return default_weather

    params = {'lat': lat, 'lon': lon, 'appid': WEATHER_API_KEY, 'units': 'imperial'} # Use imperial units for F, mph

    # IMPROVEMENT [I-10]: Use centralized retry logic
    response = _make_request_with_retries(WEATHER_API_URL, params=params, max_retries=3, timeout=10)

    if response is None:
        logging.error(f"All retries failed for weather at ({lat}, {lon}). Using default weather values.")
        return default_weather

    try:
        data = response.json()
        if 'main' in data and 'wind' in data:
            # Ensure values are floats, use defaults if keys are missing or conversion fails
            temp = float(data['main'].get('temp', default_weather['temp']))
            humidity = float(data['main'].get('humidity', default_weather['humidity']))
            pressure_hpa = float(data['main'].get('pressure', default_weather['pressure']))
            wind_speed = float(data['wind'].get('speed', default_weather['wind_speed']))
            # OpenWeatherMap wind direction 'deg' is degrees (meteorological)
            wind_dir = float(data['wind'].get('deg', default_weather['wind_dir']))

            return {
                'temp': temp, 'humidity': humidity, 'wind_speed': wind_speed,
                'wind_dir': wind_dir, 'pressure': pressure_hpa
            }
        else:
            logging.warning(f"Incomplete weather data received for ({lat}, {lon}): {data}. Using defaults.")
            return default_weather

    except json.JSONDecodeError:
        logging.warning(f"Failed to decode weather API response for ({lat}, {lon}). Using defaults.")
        return default_weather
    except (ValueError, TypeError) as e:
         logging.warning(f"Error converting weather data types for ({lat}, {lon}): {e}. Using defaults.")
         return default_weather
    except Exception as e:
        logging.error(f"Unexpected error processing weather data for ({lat}, {lon}): {e}")
        return default_weather

def add_historical_weather_features(batter_df, pitcher_df):
    """Add historical weather features using Meteostat. (Used during training)."""
    # IMPROVEMENT: Check if Meteostat library was imported successfully
    if Point is None or Daily is None:
        logging.warning("Meteostat library not available. Skipping historical weather feature addition.")
        # Add default weather columns if they don't exist
        default_weather_cols = ['temp', 'humidity', 'wind_speed', 'wind_dir', 'pressure']
        for df in [batter_df, pitcher_df]:
            if not df.empty:
                for col in default_weather_cols:
                    if col not in df.columns:
                        df[col] = LEAGUE_AVERAGES[col]
        return batter_df, pitcher_df

    logging.info("Adding historical weather features using Meteostat...")
    if batter_df.empty and pitcher_df.empty:
        logging.warning("Batter and Pitcher dataframes are empty. Skipping historical weather.")
        return batter_df, pitcher_df

    # Combine unique date/team pairs from both dataframes
    # IMPROVEMENT: Ensure 'home_team' column exists before attempting to use it
    relevant_dfs = []
    for df in [batter_df, pitcher_df]:
        if not df.empty and 'game_date' in df.columns and 'home_team' in df.columns:
            relevant_dfs.append(df[['game_date', 'home_team']])

    if not relevant_dfs:
        logging.warning("No valid DataFrames with 'game_date' and 'home_team' found. Skipping historical weather.")
        return batter_df, pitcher_df

    all_dates_teams = pd.concat(relevant_dfs).drop_duplicates().reset_index(drop=True)

    # Ensure types are correct for processing
    all_dates_teams['game_date'] = pd.to_datetime(all_dates_teams['game_date'], errors='coerce')
    all_dates_teams.dropna(subset=['game_date', 'home_team'], inplace=True)
    # IMPROVEMENT: Handle potential non-string types before applying string methods
    all_dates_teams['home_team'] = all_dates_teams['home_team'].astype(str).str.upper()

    weather_cache = {}
    default_weather = {
        'temp': LEAGUE_AVERAGES['temp'], 'humidity': LEAGUE_AVERAGES['humidity'],
        'wind_speed': LEAGUE_AVERAGES['wind_speed'], 'wind_dir': LEAGUE_AVERAGES['wind_dir'],
        'pressure': LEAGUE_AVERAGES['pressure']
    }

    # Fetch historical weather data using Meteostat
    # NOTE [I-12]: Meteostat data availability can be sparse for some locations/dates.[19, 20]
    # The current approach falls back to defaults, which is reasonable.
    for _, row in tqdm(all_dates_teams.iterrows(), total=len(all_dates_teams), desc="Fetching Historical Weather"):
        game_date, team_abbr_raw = row['game_date'], row['home_team']
        # Normalize team abbreviation using the map
        team_key = TEAM_ABBR_MAP.get(team_abbr_raw, team_abbr_raw) # Use mapped abbr, fallback to original

        # Use date object and team string as cache key
        cache_key = (game_date.date(), team_key)

        if cache_key not in weather_cache:
            coords = BALLPARK_COORDS.get(team_key, BALLPARK_COORDS.get('DEFAULT')) # Use mapped team key
            if not coords:
                 logging.warning(f"No coordinates found for team {team_key} ('{team_abbr_raw}'), using default.")
                 coords = BALLPARK_COORDS['DEFAULT']

            location = Point(coords['lat'], coords['lon'])
            start_dt = datetime.combine(game_date.date(), datetime.min.time())
            end_dt = start_dt # Fetch data for the specific day

            try:
                # Fetch daily average data [21, 22]
                data = Daily(location, start_dt, end_dt)
                data = data.fetch()

                if not data.empty:
                    daily_data = data.iloc[0]
                    # Safely get values, convert units, provide defaults if missing
                    temp_c = daily_data.get('tavg') # Average temp in Celsius
                    wind_kmh = daily_data.get('wspd') # Average wind speed in km/h
                    wind_deg = daily_data.get('wdir') # Wind direction in degrees
                    pressure_hpa = daily_data.get('pres') # Average pressure in hPa

                    # Convert units and handle missing values (NaN)
                    temp_f = (temp_c * 9/5 + 32) if pd.notna(temp_c) else default_weather['temp']
                    wind_mph = (wind_kmh * 0.621371) if pd.notna(wind_kmh) else default_weather['wind_speed']
                    wind_dir_final = wind_deg if pd.notna(wind_deg) else default_weather['wind_dir']
                    pressure_final = pressure_hpa if pd.notna(pressure_hpa) else default_weather['pressure']
                    # Meteostat Daily often lacks humidity, use default
                    humidity_final = default_weather['humidity']

                    weather_cache[cache_key] = {
                        'temp': temp_f, 'humidity': humidity_final, 'wind_speed': wind_mph,
                        'wind_dir': wind_dir_final, 'pressure': pressure_final
                    }
                else:
                    # logging.debug(f"No weather data found via Meteostat for {cache_key}, using defaults.")
                    weather_cache[cache_key] = default_weather.copy()
            except Exception as e:
                # Reduce log noise for common Meteostat station issues if needed
                logging.debug(f"Error fetching Meteostat weather for {cache_key}: {e}. Using defaults.")
                weather_cache[cache_key] = default_weather.copy() # Still use default if fetch fails

    # --- Map cached weather data back to the original DataFrames ---
    if not weather_cache:
        logging.warning("Weather cache is empty after fetching. Weather features might be defaults.")
        # Create an empty DF with expected columns if cache is empty
        weather_df = pd.DataFrame(columns=['game_date_key', 'team_key', 'temp', 'humidity',
                                           'wind_speed', 'wind_dir', 'pressure'])
    else:
        weather_df = pd.DataFrame.from_dict(weather_cache, orient='index')
        weather_df = weather_df.reset_index()
        weather_df.rename(columns={'level_0': 'game_date_key', 'level_1': 'team_key'}, inplace=True)
        # Ensure the date key is a date object for merging
        weather_df['game_date_key'] = pd.to_datetime(weather_df['game_date_key']).dt.date

    def apply_weather_mapping(df_in):
        if df_in.empty:
            return df_in
        # Ensure required columns exist before proceeding
        if 'game_date' not in df_in.columns or 'home_team' not in df_in.columns:
             logging.warning("Input DataFrame missing 'game_date' or 'home_team'. Cannot apply weather mapping.")
             # Add default weather columns if they don't exist
             for col in default_weather.keys():
                 if col not in df_in.columns:
                     df_in[col] = default_weather[col]
             return df_in

        df = df_in.copy() # Work on a copy

        # Ensure game_date is datetime and create matching keys
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df['game_date_key'] = df['game_date'].dt.date
        df['team_key_raw'] = df['home_team'].astype(str).str.upper()
        # Map team abbreviation for merging
        df['team_key'] = df['team_key_raw'].map(TEAM_ABBR_MAP).fillna(df['team_key_raw'])

        # Merge weather data using the keys
        df = pd.merge(df, weather_df, on=['game_date_key', 'team_key'], how='left')

        # Fill any weather NaNs resulting from merge mismatches with defaults
        for col in default_weather.keys():
            if col in df.columns:
                # Use fillna with the specific default value for that column
                df[col].fillna(default_weather[col], inplace=True)
            else: # Add column with default if it wasn't present at all
                df[col] = default_weather[col]

        # --- Calculate Weather-Adjusted Features ---
        # Ensure base columns exist, fill with league average if needed
        base_stat_cols = ['release_speed', 'launch_angle', 'launch_speed', 'pfx_x', 'pfx_z'] # Added more potential bases
        for col in base_stat_cols:
             if col not in df.columns:
                 df[col] = LEAGUE_AVERAGES.get(col, 0.0) # Use league avg, default to 0 if not found
             else:
                 # Ensure column is numeric before calculations
                 df[col] = safe_get_numeric(df[col], LEAGUE_AVERAGES.get(col, 0.0))

        # Example adjustments (can be refined based on domain knowledge/research)
        # Temperature adjustment on velocity (simplified)
        df['temp_adj_velo'] = df['release_speed'] * (1 + 0.001 * (df['temp'] - LEAGUE_AVERAGES['temp']))

        # Wind-adjusted launch angle (simple example, assumes wind affects angle directly)
        # Needs careful consideration of wind direction relative to field orientation (not done here)
        if 'launch_angle' in df.columns:
            # Ensure wind_dir is numeric, fill NaN with 0 (assuming North)
            wind_dir_numeric = safe_get_numeric(df['wind_dir'], 0.0)
            cos_wind_dir = np.cos(np.radians(wind_dir_numeric))
            wind_speed_numeric = safe_get_numeric(df['wind_speed'], LEAGUE_AVERAGES['wind_speed'])
            # Apply adjustment: higher wind speed in direction of hit (cos > 0) increases effective angle?
            df['wind_adj_launch'] = df['launch_angle'] * (1 + 0.002 * wind_speed_numeric * cos_wind_dir)
        else:
             df['wind_adj_launch'] = LEAGUE_AVERAGES['launch_angle'] # Default if no launch angle

        # Clean up temporary key columns
        df.drop(columns=['game_date_key', 'team_key', 'team_key_raw'], inplace=True, errors='ignore')
        return df

    batter_df_out = apply_weather_mapping(batter_df)
    pitcher_df_out = apply_weather_mapping(pitcher_df)

    logging.info("Finished adding historical weather features.")
    return batter_df_out, pitcher_df_out

# ==============================================================
# DATA LOADING & PROCESSING (MODIFIED FOR NEW SCORING)
# ==============================================================

def calculate_daily_score(df_in, is_pitcher):
    """Calculates the daily performance score (-10 to 10) based on WPA and role-specific stats."""
    if df_in.empty:
        # IMPROVEMENT: Handle empty input DataFrame gracefully
        logging.debug(f"Input DataFrame empty for {'pitcher' if is_pitcher else 'hitter'} score calculation.")
        return df_in # Return empty DF

    df = df_in.copy() # Avoid modifying the original DataFrame slice

    # --- Common Requirement: WPA ---
    # WPA ('daily_wpa') is assumed pre-calculated during aggregation based on 'delta_home_win_exp'
    # NOTE [I-16]: This sum might be inaccurate if a player changes teams/roles within the aggregation period.
    if 'daily_wpa' not in df.columns:
        logging.debug(f"Missing 'daily_wpa' column for {'pitcher' if is_pitcher else 'hitter'} score calculation. Setting WPA contribution to 0.")
        df['daily_wpa'] = 0.0
    else:
        # Ensure WPA is numeric, default to 0 if not
        df['daily_wpa'] = safe_get_numeric(df['daily_wpa'], 0.0)

    # --- Role-Specific Calculations ---
    if is_pitcher:
        # Required stats for Pitcher Score: Outs, HR, BB, HBP, K
        pitcher_req_cols = ['outs_recorded', 'home_run', 'walk', 'hit_by_pitch', 'strikeout']
        for col in pitcher_req_cols:
            if col not in df.columns:
                logging.debug(f"Missing required column '{col}' for pitcher score calculation. Filling with 0.")
                df[col] = 0
            else:
                df[col] = safe_get_numeric(df[col], 0) # Ensure columns are numeric integers/floats

        # Calculate Innings Pitched (IP)
        df['IP'] = df['outs_recorded'] / 3.0

        # Calculate FIP (Fielding Independent Pitching) [5, 6, 7, 23, 8]
        fip_constant = LEAGUE_AVERAGES['FIP_constant']
        # FIP = ((13 * HR) + (3 * (BB + HBP)) - (2 * K)) / IP + constant
        fip_numerator = (13 * df['home_run'] + 3 * (df['walk'] + df['hit_by_pitch']) - 2 * df['strikeout'])

        # FIX [I-01]: Handle IP = 0 to avoid division by zero. Assign a penalty FIP (e.g., league average + std dev, or a high fixed value).
        # Using a high fixed value (e.g., 6.00) or league avg FIP as penalty. Let's use lgFIP + 1 as penalty.
        lg_fip = LEAGUE_AVERAGES['lgFIP']
        penalty_fip = lg_fip + 1.0 # Assign slightly worse than average FIP if no outs recorded
        df['daily_fip'] = np.where(
            df['IP'] > 0.001, # Use small threshold for float comparison
            (fip_numerator / df['IP']) + fip_constant,
            penalty_fip
        )

        # Calculate WAR (Wins Above Replacement) approximation using FIP [9, 10, 11, 24]
        # Simple FanGraphs-like WAR: WAR = (lgFIP - FIP) / RunsPerWin * IP
        runs_per_win_war = LEAGUE_AVERAGES['runs_per_win']
        if runs_per_win_war <= 0: # Avoid division by zero if constant is bad
            logging.error("Invalid 'runs_per_win' constant (<=0). Cannot calculate WAR.")
            df['daily_war'] = 0.0
        else:
            # FIX [I-01]: Ensure WAR is 0 if IP is 0.
            df['daily_war'] = np.where(
                df['IP'] > 0.001,
                ((lg_fip - df['daily_fip']) / runs_per_win_war) * df['IP'],
                0.0 # Assign 0 WAR if IP is 0
            )

        # Scale components (adjust scaling factors as needed based on distribution/impact)
        # These factors significantly influence the final score and require tuning/validation.
        scaled_war = df['daily_war'] * 15 # Example scaling (WAR usually < 1 per game)
        scaled_wpa = df['daily_wpa'] * 20 # Example scaling (WPA can vary)

        # Combine scaled components into final score (ensure weights sum to 1 or adjust scale)
        # Weights: 70% WAR, 30% WPA (as per original intent inferred)
        df['daily_score'] = 0.7 * scaled_war + 0.3 * scaled_wpa

    else: # Hitters
        # Required stats for Hitter Score: PA, woba_value (summed over game)
        hitter_req_cols = ['PA', 'woba_value']
        for col in hitter_req_cols:
            if col not in df.columns:
                logging.debug(f"Missing required column '{col}' for hitter score calculation. Filling with 0.")
                df[col] = 0
            else:
                df[col] = safe_get_numeric(df[col], 0) # Ensure columns are numeric

        league_woba = LEAGUE_AVERAGES['league_woba']
        woba_scale = LEAGUE_AVERAGES['wOBA_scale']
        runs_per_win_owar = LEAGUE_AVERAGES['runs_per_win']

        # Avoid division by zero if constants are invalid
        if league_woba <= 0 or woba_scale <= 0 or runs_per_win_owar <= 0:
            logging.error("Invalid league constants (league_woba, wOBA_scale, runs_per_win <= 0). Hitter scores will be inaccurate.")
            df['daily_woba'] = 0.0
            df['daily_wrc_plus'] = 100.0 # Assign average
            df['daily_owar'] = 0.0
        else:
            # Calculate daily wOBA [1, 2]
            # wOBA = sum(woba_value) / PA
            # FIX [I-01]: Handle PA = 0. Assign 0 wOBA if no plate appearances.
            df['daily_woba'] = np.where(
                df['PA'] > 0,
                df['woba_value'] / df['PA'], # woba_value should be pre-summed from pitch data
                0.0 # Assign 0 wOBA if PA is 0
            )

            # Calculate daily wRC+ (Weighted Runs Created Plus) approximation [3, 25, 26]
            # Simple approximation: 100 * (wOBA / lgwOBA)
            # More complex formula involves wRAA, R/PA, park factors, etc. Using simple version here.
            # FIX [I-01]: Handle PA = 0. Assign average wRC+ (100) if PA=0.
            df['daily_wrc_plus'] = np.where(
                df['PA'] > 0,
                100.0 * (df['daily_woba'] / league_woba),
                100.0 # Assign league average wRC+ if PA=0
            )

            # Calculate daily oWAR (Offensive WAR) approximation [4, 27]
            # Using simple runs above average based on wOBA:
            # wRAA = ((wOBA - lgwOBA) / wOBAScale) * PA
            # WAR = wRAA / RunsPerWin
            # FIX [I-01]: Handle PA = 0. Assign 0 oWAR if PA is 0.
            df['daily_owar'] = np.where(
                df['PA'] > 0,
                (((df['daily_woba'] - league_woba) / woba_scale) * df['PA']) / runs_per_win_owar,
                0.0 # Assign 0 oWAR if PA is 0
            )

        # Scale components (adjust scaling factors)
        # These factors require tuning/validation.
        scaled_wpa = df['daily_wpa'] * 20 # Example scaling
        scaled_wrc_plus = (df['daily_wrc_plus'] - 100) / 10.0 # Center wRC+ around 0, scale
        scaled_owar = df['daily_owar'] * 50 # Example scaling (oWAR is usually small per game)

        # Combine scaled components into final score
        # Weights: 45% WPA, 35% wRC+, 20% oWAR (as per original intent inferred)
        df['daily_score'] = 0.45 * scaled_wpa + 0.35 * scaled_wrc_plus + 0.2 * scaled_owar

    # --- Final Step: Clamp and Clean Score ---
    df['daily_score'] = df['daily_score'].clip(-10, 10)
    df['daily_score'].fillna(0.0, inplace=True) # Ensure no NaNs remain

    # IMPROVEMENT: Return only necessary columns to save memory downstream
    # Keep essential identifiers and the calculated score/components
    id_col = 'pitcher' if is_pitcher else 'batter'
    keep_cols = [id_col, 'game_date', 'daily_score']
    # Add calculated stats needed for features or analysis
    if is_pitcher:
        keep_cols.extend(['daily_fip', 'daily_war', 'IP', 'outs_recorded', 'home_run', 'walk', 'hit_by_pitch', 'strikeout'])
    else:
        keep_cols.extend(['daily_woba', 'daily_wrc_plus', 'daily_owar', 'PA', 'woba_value', 'single', 'double', 'triple', 'home_run', 'walk', 'strikeout'])
    # Keep other columns if they are needed for feature engineering later (like opponent/team info)
    # Example: keep_cols.extend(['home_team', 'away_team', 'batter_opp_team', 'pitcher_opp_team'])
    # Keep base stats used for rolling features if not already included
    keep_cols.extend(['release_speed', 'release_spin_rate', 'launch_speed', 'launch_angle', 'pfx_x', 'pfx_z', 'is_barrel'])

    # Ensure all columns to keep actually exist in the dataframe before selecting
    final_cols = [col for col in keep_cols if col in df.columns]
    # Add back any original columns not explicitly mentioned, if needed
    original_cols_to_keep = [col for col in df_in.columns if col not in final_cols and col in df.columns]
    final_cols.extend(original_cols_to_keep)
    # Remove duplicates
    final_cols = list(dict.fromkeys(final_cols))

    return df[final_cols]

def load_and_process_data(player_ids=None, start_date=None, end_date=None):
    """
    Load pitch data from SQLite, aggregate game stats, calculate daily scores.
    Can optionally filter by player IDs and date range.
    """
    if player_ids is not None and not isinstance(player_ids, (list, set, tuple)):
        logging.error("player_ids must be a list, set, or tuple.")
        return pd.DataFrame(), pd.DataFrame()
    if player_ids is not None and not player_ids:
         logging.warning("load_and_process_data called with empty player_ids collection. Returning empty DataFrames.")
         return pd.DataFrame(), pd.DataFrame()

    # Format dates for SQL query
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    date_filter_sql = ""
    date_params = []
    if start_date_str and end_date_str:
        date_filter_sql = f"AND game_date BETWEEN ? AND ?"
        date_params = [start_date_str, end_date_str]
    elif start_date_str:
        date_filter_sql = f"AND game_date >= ?"
        date_params = [start_date_str]
    elif end_date_str:
        date_filter_sql = f"AND game_date <= ?"
        date_params = [end_date_str]

    player_filter_sql = ""
    player_params = []
    if player_ids:
        # IMPROVEMENT: Use parameter binding for player IDs to prevent SQL injection vulnerabilities
        # Ensure player IDs are converted to native Python types if they are numpy types
        player_ids_list = [int(pid) if isinstance(pid, np.integer) else pid for pid in player_ids]
        placeholders = ', '.join(['?'] * len(player_ids_list))
        player_filter_sql = f"AND (batter IN ({placeholders}) OR pitcher IN ({placeholders}))"
        # Player IDs need to be included twice in the parameters tuple
        player_params = tuple(player_ids_list) * 2 # Repeat for batter and pitcher check

    # Combine parameters
    sql_params = player_params + tuple(date_params)

    logging.info(f"Starting data loading and processing from DB: {DB_PATH}")
    logging.info(f"Filters: Players={bool(player_ids)}, Start={start_date_str}, End={end_date_str}")

    batter_chunks = []
    pitcher_chunks = []

    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found: {DB_PATH}")
        return pd.DataFrame(), pd.DataFrame()

    try:
        # IMPROVEMENT: Use context manager for connection
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pitch_data';")
            if not cursor.fetchone():
                logging.error("Table 'pitch_data' not found in the database.")
                return pd.DataFrame(), pd.DataFrame()

            cursor.execute("PRAGMA table_info(pitch_data);")
            available_cols_info = cursor.fetchall()
            available_cols = [info[1] for info in available_cols_info]
            # IMPROVEMENT: Check data types from PRAGMA if needed for robust type handling
            # col_types = {info[1]: info[2] for info in available_cols_info}
            logging.debug(f"Available columns in pitch_data: {available_cols}")

            required_base = ['game_date', 'batter', 'pitcher', 'events', 'inning_topbot',
                             'home_team', 'away_team', 'game_pk', 'at_bat_number']
            # woba_denom needed for accurate PA calculation
            required_scoring = ['delta_home_win_exp', 'woba_value', 'woba_denom', 'description']
            optional_stats = [
                'estimated_woba_using_speedangle', 'estimated_ba_using_speedangle',
                'launch_speed', 'launch_angle', 'release_speed', 'release_spin_rate', # Ensure consistent naming
                'zone', 'pfx_x', 'pfx_z', 'hc_x', 'hc_y', 'pitch_number', 'pitch_type',
                'outs_when_up', 'is_barrel' # Include pre-calculated barrel if exists
            ]
            # Ensure 'release_spin_rate' is used if available, map from old names if necessary
            if 'release_spin_rate' not in available_cols and 'spin_rate' in available_cols:
                 optional_stats.append('spin_rate') # Add old name if new one isn't there
                 logging.debug("Using 'spin_rate' as 'release_spin_rate' not found.")

            select_cols = list(set(required_base + required_scoring + optional_stats) & set(available_cols))

            missing_essentials = [col for col in required_base + required_scoring if col not in select_cols]
            if missing_essentials:
                # Allow missing 'woba_denom' if PA fallback logic is acceptable
                if 'woba_denom' in missing_essentials and len(missing_essentials) == 1:
                    logging.warning("Missing 'woba_denom' column. Will infer Plate Appearances (PA) from events (less accurate).")
                    if 'events' not in select_cols: select_cols.append('events') # Ensure events is selected for fallback
                    select_cols = list(set(select_cols)) # Remove duplicates
                    # Remove woba_denom from missing_essentials list as it's handled
                    missing_essentials.remove('woba_denom')

                # If other essentials are still missing, abort
                if missing_essentials:
                    logging.error(f"Missing essential columns in DB: {missing_essentials}. Cannot proceed.")
                    return pd.DataFrame(), pd.DataFrame()

            # Rename 'spin_rate' to 'release_spin_rate' in selection if needed
            select_cols_final = select_cols.copy() # Work with a copy
            select_cols_str_list = []
            has_spin_rate = 'spin_rate' in select_cols_final
            has_release_spin_rate = 'release_spin_rate' in select_cols_final

            for col in select_cols_final:
                 if col == 'spin_rate' and not has_release_spin_rate:
                      select_cols_str_list.append(f'"{col}" AS "release_spin_rate"')
                      logging.debug("Mapping DB column 'spin_rate' to 'release_spin_rate'.")
                 elif col == 'release_spin_rate' and has_spin_rate:
                      # If both exist, prefer 'release_spin_rate', don't select 'spin_rate' separately
                      select_cols_str_list.append(f'"{col}"')
                 elif col == 'spin_rate' and has_release_spin_rate:
                      # If both exist and we are processing 'spin_rate', skip it as 'release_spin_rate' is preferred
                      continue
                 else:
                      select_cols_str_list.append(f'"{col}"') # Quote column names

            select_cols_str = ', '.join(select_cols_str_list)

            logging.debug(f"Selected columns for query: {select_cols_str}")

            # NOTE [I-04]: Using game_date, game_pk, at_bat_number, pitch_number provides robust ordering.
            order_cols_options = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
            order_cols = [col for col in order_cols_options if col in select_cols]
            if not order_cols:
                 logging.warning("Essential ordering columns (game_pk, at_bat_number, pitch_number) missing. Relying on game_date.")
                 order_cols = ['game_date'] # Fallback to game_date only if others missing
            # IMPROVEMENT: Ensure rowid is not needed if proper keys exist
            # if not order_cols: order_cols = ['rowid']
            logging.debug(f"Ordering data by: {order_cols}")

            # IMPROVEMENT: Use parameter binding in main query and count query
            base_query_where = "WHERE game_date IS NOT NULL AND batter IS NOT NULL AND pitcher IS NOT NULL AND events IS NOT NULL"
            query = f"SELECT {select_cols_str} FROM pitch_data {base_query_where} {player_filter_sql} {date_filter_sql} ORDER BY {', '.join(f'"{c}"' for c in order_cols)}" # Quote order columns
            count_query = f"SELECT COUNT(*) FROM pitch_data {base_query_where} {player_filter_sql} {date_filter_sql}"

            # Estimate total rows for progress bar
            total_rows = 0
            n_chunks = None
            try:
                # Use parameters with count query
                count_cursor = conn.cursor()
                count_cursor.execute(count_query, sql_params)
                result = count_cursor.fetchone()
                total_rows = result[0] if result else 0
                count_cursor.close()

                if total_rows == 0:
                    logging.warning(f"No rows found matching filters in DB.")
                    # logging.debug(f"Count Query: {count_query} | Params: {sql_params}")
                    return pd.DataFrame(), pd.DataFrame()
                n_chunks = max(1, (total_rows // CHUNKSIZE) + (1 if total_rows % CHUNKSIZE > 0 else 0))
                logging.info(f"Estimated rows to process: {total_rows} in {n_chunks} chunks.")
            except sqlite3.Error as e:
                 logging.warning(f"SQLite error during row count: {e}. Progress bar may be inaccurate.")
            except Exception as e:
                logging.warning(f"Could not estimate total rows: {e}. Progress bar may be inaccurate.")

            # Process data in chunks using parameters [28, 29]
            try:
                chunk_iterator = pd.read_sql_query(query, conn, params=sql_params, chunksize=CHUNKSIZE)
            except Exception as e:
                 logging.error(f"Error executing main SQL query: {e}")
                 # logging.debug(f"Failed Query: {query} | Params: {sql_params}")
                 return pd.DataFrame(), pd.DataFrame()

            for batch_num, batch in enumerate(tqdm(chunk_iterator, total=n_chunks, desc="Processing DB Chunks")):
                if batch.empty: continue
                # IMPROVEMENT: Explicitly copy batch to avoid SettingWithCopyWarning later
                df = batch.copy()

                # --- Basic Type Conversions & Cleaning ---
                df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
                for id_col in ['batter', 'pitcher']:
                    # Use safe_get_numeric and cast to nullable Int64 for flexibility
                    df[id_col] = pd.to_numeric(df[id_col], errors='coerce') # Coerce first
                    df[id_col] = df[id_col].astype('Int64') # Use nullable integer type

                # Drop rows essential columns are missing AFTER type conversion
                df.dropna(subset=['game_date', 'batter', 'pitcher', 'events', 'inning_topbot'], inplace=True)
                if df.empty: continue

                # --- Calculate PA per Pitch Event ---
                if 'woba_denom' in df.columns:
                    # Ensure woba_denom is numeric before comparison
                    df['is_pa_outcome'] = (safe_get_numeric(df['woba_denom'], 0) == 1).astype(int)
                else:
                    # Fallback: Infer PA from event types (less accurate)
                    pa_ending_events = [
                        'strikeout', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run',
                        'field_out', 'sac_fly', 'sac_bunt', 'force_out', 'grounded_into_double_play',
                        'fielders_choice', 'field_error', 'double_play', 'triple_play', 'batter_interference',
                        'catcher_interf' # Added catcher interference
                    ]
                    df['is_pa_outcome'] = df['events'].astype(str).isin(pa_ending_events).astype(int)

                # --- Calculate Outs per Event ---
                # IMPROVEMENT: More robust outs calculation based on event types and descriptions
                df['outs_made'] = 0
                if 'events' in df.columns:
                    # Map standard events to outs
                    event_out_map = {
                        'strikeout': 1, 'field_out': 1, 'sac_fly': 1, 'sac_bunt': 1, 'force_out': 1,
                        'grounded_into_double_play': 2, 'double_play': 2, 'triple_play': 3,
                        'fielders_choice_out': 1, 'caught_stealing_2b': 1, 'caught_stealing_3b': 1, 'caught_stealing_home': 1,
                        'pickoff_1b': 1, 'pickoff_2b': 1, 'pickoff_3b': 1, 'pickoff_caught_stealing_2b': 1,
                        'pickoff_caught_stealing_3b': 1, 'pickoff_caught_stealing_home': 1, 'other_out': 1,
                        'sac_fly_double_play': 2, 'sac_bunt_double_play': 2, 'strikeout_double_play': 2
                    }
                    # Ensure events column is string before mapping
                    df['outs_made'] = df['events'].astype(str).map(event_out_map).fillna(0)

                    # Refine based on description (handle potential NaNs in description)
                    if 'description' in df.columns:
                        desc_lower = df['description'].astype(str).str.lower()
                        df.loc[desc_lower.str.contains('strikes out', na=False), 'outs_made'] = np.maximum(df['outs_made'], 1)
                        df.loc[desc_lower.str.contains('double play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 2)
                        df.loc[desc_lower.str.contains('triple play', na=False), 'outs_made'] = np.maximum(df['outs_made'], 3)

                    df['outs_made'] = df['outs_made'].astype(int)

                # --- Event Flags (ensure events column is string) ---
                event_flags = { 'single': 'single', 'double': 'double', 'triple': 'triple', 'home_run': 'home_run',
                                'walk': 'walk', 'hit_by_pitch': 'hit_by_pitch', 'strikeout': 'strikeout'}
                if 'events' in df.columns:
                    events_str = df['events'].astype(str)
                    for event, flag_col in event_flags.items():
                        df[flag_col] = (events_str == event).astype(int)
                    # Refine strikeout using description if available
                    if 'description' in df.columns:
                        desc_lower = df['description'].astype(str).str.lower()
                        df.loc[desc_lower.str.contains('strikes out', na=False), 'strikeout'] = 1
                else: # Set flags to 0 if 'events' is missing
                    for flag_col in event_flags.values(): df[flag_col] = 0

                # --- Opponent Teams ---
                # Ensure inning_topbot is string before comparison
                inning_str = df['inning_topbot'].astype(str)
                df['batter_opp_team'] = np.where(inning_str == 'Top', df['home_team'], df['away_team'])
                df['pitcher_opp_team'] = np.where(inning_str == 'Top', df['away_team'], df['home_team'])

                # --- Numeric Conversions & Default Fills ---
                # Use a copy to avoid modifying the original LEAGUE_AVERAGES
                default_vals = LEAGUE_AVERAGES.copy()
                # Add defaults for columns used in calculations or features if not in LEAGUE_AVERAGES
                default_vals.update({'delta_home_win_exp': 0.0,
                                     'woba_value': LEAGUE_AVERAGES.get('league_woba', 0.315), # Default woba_value to league avg
                                     'hc_x': 125.0, 'hc_y': 150.0, 'zone': 14.0,
                                     'pitch_number': 1.0, 'outs_when_up': 0.0,
                                     'release_spin_rate': LEAGUE_AVERAGES.get('spin_rate', 2400.0)}) # Use spin_rate default if needed
                for col, default in default_vals.items():
                    if col in df.columns:
                        df[col] = safe_get_numeric(df[col], default)
                    # else: # Avoid adding columns not present in the data
                    #     logging.debug(f"Column '{col}' not found in chunk, skipping default fill.")

                # --- Derived Features (Barrel) ---
                # Calculate only if 'is_barrel' is not present or has NaNs after loading
                calculate_barrel = False
                if 'is_barrel' not in df.columns:
                    calculate_barrel = True
                elif df['is_barrel'].isnull().any():
                     df['is_barrel'] = safe_get_numeric(df['is_barrel'], 0) # Fill existing NaNs first
                     if df['is_barrel'].isnull().any(): # Check again if safe_get_numeric failed
                         calculate_barrel = True

                if calculate_barrel:
                    if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
                        # Ensure inputs are numeric
                        ls = safe_get_numeric(df['launch_speed'], 0.0)
                        la = safe_get_numeric(df['launch_angle'], 0.0)
                        # Barrel definition (example, can be refined)
                        # Speed >= 98 mph AND Angle between 26-30 deg
                        # OR Speed >= 100 mph AND Angle between 24-33 deg (example refinement)
                        # Using the simpler definition from original script for consistency:
                        ls_cond = ls >= 98
                        # Dynamic angle range based on speed (example from Statcast)
                        la_min = np.select(
                            [ls >= 116, ls >= 110, ls >= 105, ls >= 100, ls >= 98],
                            [8.0, 12.0, 16.0, 20.0, 26.0],
                            default=-np.inf # Condition for ls < 98
                        )
                        la_max = np.select(
                            [ls >= 116, ls >= 110, ls >= 105, ls >= 100, ls >= 98],
                            [33.0, 36.0, 39.0, 44.0, 50.0],
                            default=np.inf # Condition for ls < 98
                        )

                        la_cond = (la >= la_min) & (la <= la_max)
                        # Ensure launch speed meets minimum threshold
                        barrel_mask = (ls >= 98) & la_cond
                        df['is_barrel'] = barrel_mask.astype(int)
                    else:
                        logging.debug("Missing 'launch_speed' or 'launch_angle'. Cannot calculate 'is_barrel'. Setting to 0.")
                        df['is_barrel'] = 0
                else: # Ensure existing column is integer
                    df['is_barrel'] = df['is_barrel'].astype(int)

                # --- Aggregation Dictionary ---
                # IMPROVEMENT: Define aggregations dynamically based on available columns
                agg_dict_base = {
                    # Counting stats
                    'single': 'sum', 'double': 'sum', 'triple': 'sum', 'home_run': 'sum',
                    'walk': 'sum', 'hit_by_pitch': 'sum', 'strikeout': 'sum',
                    'is_barrel': 'sum',
                    # Value sums
                    'woba_value': 'sum',
                    'delta_home_win_exp': 'sum', # For daily WPA
                    # PA and Outs
                    'is_pa_outcome': 'sum', # To get Plate Appearances (PA)
                    'outs_made': 'sum', # To get Outs Recorded (pitchers)
                    # Averaged stats (use mean, handle NaNs within mean)
                    'estimated_woba_using_speedangle': 'mean', 'estimated_ba_using_speedangle': 'mean',
                    'release_speed': 'mean', 'release_spin_rate': 'mean',
                    'launch_speed': 'mean', 'launch_angle': 'mean',
                    'pfx_x': 'mean', 'pfx_z': 'mean',
                    # Add first() for columns needed later but not aggregated numerically
                    'home_team': 'first', 'away_team': 'first',
                    'batter_opp_team': 'first', # Keep opponent info if needed
                    'pitcher_opp_team': 'first'
                }
                # Filter dict to only include columns present in the current chunk
                agg_dict = {k: v for k, v in agg_dict_base.items() if k in df.columns}

                # Ensure essential aggregation columns exist
                essential_agg_cols = ['is_pa_outcome', 'outs_made', 'woba_value', 'delta_home_win_exp']
                missing_essentials_agg = [col for col in essential_agg_cols if col not in agg_dict]
                if missing_essentials_agg:
                     logging.warning(f"Chunk {batch_num + 1} missing essential columns for aggregation ({missing_essentials_agg}). Skipping aggregation for this chunk.")
                     continue
                if not agg_dict:
                    logging.warning(f"Chunk {batch_num + 1} has no columns to aggregate. Skipping.")
                    continue

                # --- Perform Aggregations ---
                try:
                    # Group by game date and player ID
                    # NOTE [I-03]: This aggregation level loses per-game opponent context if a player plays multiple games on one day (rare but possible)
                    # or if opponent features need finer granularity.
                    group_cols_batter = ['game_date', 'batter']
                    # Use observed=True for potential performance benefits with categorical-like IDs
                    batter_agg = df.groupby(group_cols_batter, as_index=False, observed=True).agg(agg_dict)
                    batter_agg.rename(columns={'is_pa_outcome': 'PA', 'delta_home_win_exp': 'daily_wpa'}, inplace=True)
                    # Apply score calculation
                    batter_df_processed = calculate_daily_score(batter_agg, is_pitcher=False)
                    if not batter_df_processed.empty:
                        batter_chunks.append(batter_df_processed)

                    group_cols_pitcher = ['game_date', 'pitcher']
                    pitcher_agg = df.groupby(group_cols_pitcher, as_index=False, observed=True).agg(agg_dict)
                    pitcher_agg.rename(columns={'outs_made': 'outs_recorded', 'delta_home_win_exp': 'daily_wpa'}, inplace=True)
                    # Apply score calculation
                    pitcher_df_processed = calculate_daily_score(pitcher_agg, is_pitcher=True)
                    if not pitcher_df_processed.empty:
                        pitcher_chunks.append(pitcher_df_processed)

                    # Explicitly delete intermediate dataframes and collect garbage
                    del df, batter_agg, pitcher_agg, batter_df_processed, pitcher_df_processed
                    gc.collect()

                except Exception as e:
                    logging.error(f"Error during aggregation in chunk {batch_num + 1}: {e}")
                    traceback.print_exc()
                    # Optionally, save problematic chunk for debugging:
                    # df.to_csv(f"debug_chunk_{batch_num+1}.csv")
                    continue # Continue to next chunk

            # --- Concatenate Chunks ---
            batter_df = pd.concat(batter_chunks, ignore_index=True) if batter_chunks else pd.DataFrame()
            pitcher_df = pd.concat(pitcher_chunks, ignore_index=True) if pitcher_chunks else pd.DataFrame()

            # Final check for data types after concatenation
            if not batter_df.empty:
                batter_df['batter'] = batter_df['batter'].astype('Int64')
                batter_df['game_date'] = pd.to_datetime(batter_df['game_date']) # Ensure datetime type
            if not pitcher_df.empty:
                pitcher_df['pitcher'] = pitcher_df['pitcher'].astype('Int64')
                pitcher_df['game_date'] = pd.to_datetime(pitcher_df['game_date']) # Ensure datetime type

    except sqlite3.Error as e:
        logging.error(f"SQLite error during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error during data loading: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

    logging.info("Data loading and daily score calculation complete.")
    logging.info(f"Processed data - Batters: {len(batter_df)}, Pitchers: {len(pitcher_df)}")
    return batter_df, pitcher_df

# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

def create_features(df, is_pitcher):
    """Create rolling and lagged features for the model using historical performance."""
    player_type = 'pitchers' if is_pitcher else 'batters'
    logging.info(f"Creating features for {player_type}...")
    if df.empty:
        logging.warning(f"{player_type.capitalize()} DataFrame is empty. Skipping feature creation.")
        return pd.DataFrame(), [] # Return empty DF and empty feature list

    player_id_col = 'pitcher' if is_pitcher else 'batter'

    # Ensure essential columns exist
    if player_id_col not in df.columns or 'game_date' not in df.columns:
        logging.error(f"Input DataFrame for {player_type} missing '{player_id_col}' or 'game_date'. Cannot create features.")
        return pd.DataFrame(), []

    # Sort by player and date for correct rolling calculations
    # IMPROVEMENT: Ensure stable sort order, handle potential NaNs in sort keys
    sort_keys = [player_id_col, 'game_date']
    # Add PA/IP for tie-breaking within a date if needed, check existence first
    sort_key_secondary = 'PA' if not is_pitcher and 'PA' in df.columns else ('IP' if is_pitcher and 'IP' in df.columns else None)
    if sort_key_secondary:
        sort_keys.append(sort_key_secondary)
        df[sort_key_secondary] = safe_get_numeric(df[sort_key_secondary], 0) # Ensure numeric for sort

    # Ensure game_date is datetime
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    # Ensure player ID is consistently numeric for grouping and sorting
    df[player_id_col] = pd.to_numeric(df[player_id_col], errors='coerce')
    # Drop rows where essential sort keys are NaN
    df.dropna(subset=[player_id_col, 'game_date'], inplace=True)
    df[player_id_col] = df[player_id_col].astype(int) # Cast to standard int after dropna

    df = df.sort_values(by=sort_keys, ascending=True)

    # --- Feature Definitions ---
    # Define base columns that might be used for rolling features
    # Include calculated stats and potentially raw counts/averages from aggregation
    stats_to_roll = ['daily_score', 'PA', 'strikeout', 'walk', 'home_run', 'is_barrel',
                     'release_speed', 'launch_speed', 'launch_angle', 'release_spin_rate',
                     'pfx_x', 'pfx_z'] # Add relevant stats from aggregation/calculation
    if is_pitcher:
        stats_to_roll.extend(['outs_recorded', 'daily_fip', 'daily_war', 'IP'])
    else: # Hitter specific
        stats_to_roll.extend(['daily_woba', 'daily_wrc_plus', 'daily_owar'])

    # Filter stats_to_roll to only include columns actually present in the DataFrame
    available_stats_to_roll = [stat for stat in stats_to_roll if stat in df.columns]
    if not available_stats_to_roll:
         logging.warning(f"No columns available for rolling feature calculation for {player_type}.")
         # Return df with no new features, but maybe add opponent placeholder
         df['opp_team_score_allowed'] = 0.0 # Placeholder
         return df, ['opp_team_score_allowed']

    logging.debug(f"Columns available for rolling features ({player_type}): {available_stats_to_roll}")

    # Define rolling window sizes
    rolling_windows = [N_GAMES, 30] # e.g., 15-game and 30-game rolling averages

    # --- Group by player and calculate features ---
    # IMPROVEMENT: Use transform for potentially faster rolling calculations within groups
    # However, groupby().apply() is more flexible if complex logic is needed per group. Sticking with groupby().apply() for clarity here.

    all_features_list = []
    # Use observed=True to avoid issues with categorical indices if player_id was ever categorical
    # Use group_keys=False to prevent adding the group key as an index level
    grouped = df.groupby(player_id_col, group_keys=False, observed=True)

    # Define a function to apply to each group
    def calculate_group_features(group):
        # group = group.sort_values(by='game_date') # Already sorted before groupby

        # Calculate rolling averages/sums for defined stats
        for window in rolling_windows:
            # Use a smaller min_periods for shorter windows, ensure it's at least 1
            min_p = max(1, window // 3)
            for stat in available_stats_to_roll:
                # Ensure stat column is numeric before rolling
                if pd.api.types.is_numeric_dtype(group[stat]):
                    # Calculate rolling mean, shift by 1 to prevent data leakage (use past N games) [30, 31, 32]
                    feature_name_mean = f'roll_{window}g_{stat}_mean'
                    group[feature_name_mean] = group[stat].rolling(window=window, min_periods=min_p).mean().shift(1)
                    # Calculate rolling sum for counting stats
                    if stat in ['PA', 'strikeout', 'walk', 'home_run', 'is_barrel', 'outs_recorded', 'single', 'double', 'triple']:
                        feature_name_sum = f'roll_{window}g_{stat}_sum'
                        group[feature_name_sum] = group[stat].rolling(window=window, min_periods=min_p).sum().shift(1)
                    # Add rolling std dev for key performance indicators?
                    if stat in ['daily_score', 'daily_fip', 'daily_woba']:
                         feature_name_std = f'roll_{window}g_{stat}_std'
                         group[feature_name_std] = group[stat].rolling(window=window, min_periods=min_p).std().shift(1)
                else:
                    logging.warning(f"Column '{stat}' in group for player {group[player_id_col].iloc[0]} is not numeric. Skipping rolling features for it.")

        # Shift features to create simple lagged features (previous game's stats)
        lagged_stats = ['daily_score'] # Add other key stats if needed (e.g., 'PA', 'IP')
        for stat in lagged_stats:
             if stat in group.columns and pd.api.types.is_numeric_dtype(group[stat]):
                 group[f'prev_{stat}'] = group[stat].shift(1)

        return group

    # Apply the function to each group
    try:
        df = grouped.apply(calculate_group_features)
        # Reset index to ensure player_id_col is a column, not an index
        df = df.reset_index(drop=True)
    except Exception as e:
        logging.error(f"Error during feature calculation for {player_type}: {e}")
        traceback.print_exc()
        return pd.DataFrame(), []

    # Update all_features_list with newly created columns
    all_features_list = [col for col in df.columns if col.startswith(('roll_', 'prev_'))]
    logging.debug(f"Generated features for {player_type}: {all_features_list}")

    # --- Opponent Team Features ---
    # IMPROVEMENT: Add opponent team performance (e.g., average score allowed by opposing team)
    opp_team_col = 'pitcher_opp_team' if is_pitcher else 'batter_opp_team'
    if opp_team_col in df.columns:
        # Calculate league-wide team performance (runs/score allowed) based on available data
        # Use daily_score as proxy for performance
        team_score_allowed = df.groupby(['game_date', opp_team_col])['daily_score'].mean().reset_index()
        team_score_allowed = team_score_allowed.groupby(opp_team_col)['daily_score'].rolling(
            window=N_GAMES, min_periods=1
        ).mean().shift(1).reset_index() # Shift to avoid leakage
        team_score_allowed = team_score_allowed.groupby('level_1').last().reset_index()
        team_score_allowed = team_score_allowed.rename(columns={'daily_score': 'opp_team_score_allowed', opp_team_col: 'opp_team'})
        # Merge back to main dataframe
        df = df.merge(
            team_score_allowed[['opp_team', 'opp_team_score_allowed']],
            left_on=opp_team_col,
            right_on='opp_team',
            how='left'
        )
        df['opp_team_score_allowed'].fillna(0.0, inplace=True) # Fill missing with neutral value
        all_features_list.append('opp_team_score_allowed')
        df.drop(columns=['opp_team'], inplace=True, errors='ignore')

    # --- Weather Features ---
    # Weather features (temp, humidity, etc.) are assumed to be in the DataFrame from load_and_process_data
    # If not present, they should be added upstream (e.g., during historical data processing or prediction)
    weather_features = ['temp', 'temp_adj_velo', 'wind_adj_launch']
    for wf in weather_features:
        if wf in df.columns:
            all_features_list.append(wf)
        else:
            logging.debug(f"Weather feature '{wf}' not found in {player_type} DataFrame.")

    # --- Fill Missing Values in Features ---
    for feature in all_features_list:
        if feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Use median or mean of the player's own historical data if enough data points
                df[feature] = df.groupby(player_id_col)[feature].transform(
                    lambda x: x.fillna(x.median() if x.notna().sum() >= 3 else LEAGUE_AVERAGES.get(feature, 0.0))
                )
                # Final fallback to league average or 0
                df[feature].fillna(LEAGUE_AVERAGES.get(feature, 0.0), inplace=True)
            else:
                logging.warning(f"Feature '{feature}' is non-numeric in {player_type} DataFrame. Dropping.")
                df.drop(columns=[feature], inplace=True)
                all_features_list.remove(feature)

    # --- Ensure Feature Consistency ---
    # Drop any feature columns with all NaNs or non-numeric types after filling
    final_features = []
    for feature in all_features_list:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]) and df[feature].notna().any():
            final_features.append(feature)
        else:
            logging.debug(f"Dropping feature '{feature}' from {player_type} due to invalid data or all NaNs.")
            if feature in df.columns:
                df.drop(columns=[feature], inplace=True)

    logging.info(f"Feature creation complete for {player_type}. Total features: {len(final_features)}")
    return df, final_features

# ==============================================================
# MODEL LOADING & PREDICTION
# ==============================================================

def load_model_and_features(is_pitcher):
    """Load the trained model and feature list for batters or pitchers."""
    model_path = PITCHER_MODEL_PATH if is_pitcher else BATTER_MODEL_PATH
    features_path = PITCHER_FEATURES_PATH if is_pitcher else BATTER_FEATURES_PATH
    player_type = 'pitcher' if is_pitcher else 'batter'

    model = None
    feature_list = []

    # Load model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logging.info(f"Loaded {player_type} model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading {player_type} model from {model_path}: {e}")
            model = None
    else:
        logging.error(f"{player_type.capitalize()} model file not found at {model_path}")

    # Load feature list
    if os.path.exists(features_path):
        try:
            with open(features_path, 'rb') as f:
                feature_list = pickle.load(f)
            logging.info(f"Loaded {player_type} features from {features_path}")
        except Exception as e:
            logging.error(f"Error loading {player_type} features from {features_path}: {e}")
            feature_list = []
    else:
        logging.error(f"{player_type.capitalize()} feature list file not found at {features_path}")

    return model, feature_list

def predict_for_players(df, model, feature_list, is_pitcher, weather_data, scheduled_games):
    """Generate predictions for players using the loaded model with weather data."""
    player_type = 'pitchers' if is_pitcher else 'batters'
    player_id_col = 'pitcher' if is_pitcher else 'batter'
    logging.info(f"Generating predictions for {player_type}...")

    if df.empty:
        logging.warning(f"{player_type.capitalize()} DataFrame is empty. No predictions will be generated.")
        return pd.DataFrame()

    if not model or not feature_list:
        logging.error(f"Cannot predict for {player_type}: Model or feature list is missing.")
        return pd.DataFrame()

    # Filter for players in scheduled games
    scheduled_player_ids = set()
    for game in scheduled_games:
        home_pitcher = game.get('home_pitcher')
        away_pitcher = game.get('away_pitcher')
        if is_pitcher:
            if home_pitcher:
                scheduled_player_ids.add(home_pitcher)
            if away_pitcher:
                scheduled_player_ids.add(away_pitcher)
        # For batters, we don't have lineup data, so process all recent batters
        # Alternatively, could filter by team roster if available

    if is_pitcher:
        # Only keep data for scheduled pitchers
        df = df[df[player_id_col].isin(scheduled_player_ids)]
        if df.empty:
            logging.warning(f"No {player_type} in DataFrame match scheduled games. No predictions.")
            return pd.DataFrame()

    # Create game-to-weather mapping
    game_mapping = {}
    for game in scheduled_games:
        game_pk = game.get('game_pk')
        home_team = game.get('home_team_abbr')
        weather = weather_data.get(home_team, LEAGUE_AVERAGES)
        game_mapping[game_pk] = {
            'home_team': home_team,
            'weather': weather
        }

    # Assign weather features based on game
    weather_features = ['temp', 'humidity', 'wind_speed', 'wind_dir', 'pressure']
    for feat in weather_features:
        df[feat] = LEAGUE_AVERAGES.get(feat, 0.0)  # Initialize with defaults

    # Map weather data to players based on game
    if 'game_pk' in df.columns:
        for idx, row in df.iterrows():
            game_pk = row['game_pk']
            game_info = game_mapping.get(game_pk)
            if game_info:
                for feat in weather_features:
                    df.at[idx, feat] = game_info['weather'].get(feat, LEAGUE_AVERAGES.get(feat, 0.0))
            else:
                logging.debug(f"No game info for game_pk {game_pk}. Using default weather.")
    else:
        logging.warning(f"'game_pk' column missing in {player_type} DataFrame. Using default weather.")
        for feat in weather_features:
            df[feat] = LEAGUE_AVERAGES.get(feat, 0.0)

    # Recalculate weather-adjusted features
    base_stat_cols = ['release_speed', 'launch_angle', 'launch_speed', 'pfx_x', 'pfx_z']
    for col in base_stat_cols:
        if col not in df.columns:
            df[col] = LEAGUE_AVERAGES.get(col, 0.0)
        else:
            df[col] = safe_get_numeric(df[col], LEAGUE_AVERAGES.get(col, 0.0))

    df['temp_adj_velo'] = df['release_speed'] * (1 + 0.001 * (df['temp'] - LEAGUE_AVERAGES['temp']))
    if 'launch_angle' in df.columns:
        wind_dir_numeric = safe_get_numeric(df['wind_dir'], 0.0)
        cos_wind_dir = np.cos(np.radians(wind_dir_numeric))
        wind_speed_numeric = safe_get_numeric(df['wind_speed'], LEAGUE_AVERAGES['wind_speed'])
        df['wind_adj_launch'] = df['launch_angle'] * (1 + 0.002 * wind_speed_numeric * cos_wind_dir)
    else:
        df['wind_adj_launch'] = LEAGUE_AVERAGES['launch_angle']

    # Ensure all model features are present
    missing_features = [f for f in feature_list if f not in df.columns]
    for feature in missing_features:
        df[feature] = LEAGUE_AVERAGES.get(feature, 0.0)
        logging.warning(f"Feature '{feature}' missing in {player_type} data. Filled with default value.")

    # Select only the features needed by the model
    X = df[feature_list].copy()
    for col in X.columns:
        X[col] = safe_get_numeric(X[col], LEAGUE_AVERAGES.get(col, 0.0))

    # Handle any remaining NaNs
    X.fillna(0.0, inplace=True)

    # Generate predictions
    try:
        predictions = model.predict(X)
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            # Assuming positive class is index 1
            predictions_proba = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        else:
            predictions_proba = predictions  # Fallback to raw predictions
    except Exception as e:
        logging.error(f"Error during prediction for {player_type}: {e}")
        return pd.DataFrame()

    # Create output DataFrame
    output_df = df[[player_id_col, 'game_date']].copy()
    output_df['predicted_score'] = predictions
    output_df['prediction_confidence'] = predictions_proba
    if 'home_team' in df.columns and 'away_team' in df.columns:
        output_df['home_team'] = df['home_team']
        output_df['away_team'] = df['away_team']
    if 'game_pk' in df.columns:
        output_df['game_pk'] = df['game_pk']

    logging.info(f"Generated predictions for {len(output_df)} {player_type}.")
    return output_df

def predict_top_performers(prediction_date):
    """Predict top-performing batters and pitchers for a given date."""
    logging.info(f"Starting prediction for top performers on {prediction_date.strftime('%Y-%m-%d')}")

    # Fetch schedule and player IDs
    scheduled_games, all_player_ids = get_schedule_for_date(prediction_date)
    if not scheduled_games:
        logging.warning(f"No games scheduled for {prediction_date.strftime('%Y-%m-%d')}. No predictions possible.")
        return {'batters': [], 'pitchers': []}

    # Fetch weather data for home teams
    weather_data = {}
    for game in scheduled_games:
        home_team = game.get('home_team_abbr')
        venue_name = game.get('venue_name', 'Unknown')
        if not home_team:
            logging.warning(f"No home team specified for game {game.get('game_pk')}. Skipping weather fetch.")
            continue
        team_key = TEAM_ABBR_MAP.get(home_team, home_team)
        coords = BALLPARK_COORDS.get(team_key, BALLPARK_COORDS.get('DEFAULT'))
        if not coords:
            logging.warning(f"No coordinates for team {team_key} (venue: {venue_name}). Using default weather.")
            weather_data[home_team] = LEAGUE_AVERAGES.copy()
            continue
        weather = get_openweathermap_weather(coords['lat'], coords['lon'])
        weather_data[home_team] = weather
        logging.debug(f"Weather for {home_team} (venue: {venue_name}): {weather}")

    # Load historical data
    start_date = prediction_date - timedelta(days=HISTORICAL_DAYS_FOR_FEATURES)
    batter_df, pitcher_df = load_and_process_data(
        player_ids=all_player_ids if all_player_ids else None,
        start_date=start_date,
        end_date=prediction_date - timedelta(days=1)  # Exclude prediction date
    )

    # Add historical weather features (for feature engineering)
    batter_df, pitcher_df = add_historical_weather_features(batter_df, pitcher_df)

    # Load models and features
    batter_model, batter_features = load_model_and_features(is_pitcher=False)
    pitcher_model, pitcher_features = load_model_and_features(is_pitcher=True)

    # Create features
    batter_df, batter_feature_list = create_features(batter_df, is_pitcher=False)
    pitcher_df, pitcher_feature_list = create_features(pitcher_df, is_pitcher=True)

    # Generate predictions
    batter_predictions = predict_for_players(
        batter_df, batter_model, batter_features, is_pitcher=False, weather_data=weather_data, scheduled_games=scheduled_games
    )
    pitcher_predictions = predict_for_players(
        pitcher_df, pitcher_model, pitcher_features, is_pitcher=True, weather_data=weather_data, scheduled_games=scheduled_games
    )

    # Select top performers
    top_batters = []
    top_pitchers = []

    if not batter_predictions.empty:
        batter_predictions = batter_predictions.sort_values(by='predicted_score', ascending=False).head(TOP_N)
        for _, row in batter_predictions.iterrows():
            top_batters.append({
                'player_id': int(row['batter']) if pd.notna(row['batter']) else None,
                'predicted_score': float(row['predicted_score']) if pd.notna(row['predicted_score']) else 0.0,
                'confidence': float(row['prediction_confidence']) if pd.notna(row['prediction_confidence']) else 0.0,
                'game_date': row['game_date'].isoformat() if pd.notna(row['game_date']) else prediction_date.isoformat(),
                'home_team': str(row['home_team']) if 'home_team' in row and pd.notna(row['home_team']) else 'Unknown',
                'away_team': str(row['away_team']) if 'away_team' in row and pd.notna(row['away_team']) else 'Unknown',
                'game_pk': int(row['game_pk']) if 'game_pk' in row and pd.notna(row['game_pk']) else None
            })

    if not pitcher_predictions.empty:
        pitcher_predictions = pitcher_predictions.sort_values(by='predicted_score', ascending=False).head(TOP_N)
        for _, row in pitcher_predictions.iterrows():
            top_pitchers.append({
                'player_id': int(row['pitcher']) if pd.notna(row['pitcher']) else None,
                'predicted_score': float(row['predicted_score']) if pd.notna(row['predicted_score']) else 0.0,
                'confidence': float(row['prediction_confidence']) if pd.notna(row['prediction_confidence']) else 0.0,
                'game_date': row['game_date'].isoformat() if pd.notna(row['game_date']) else prediction_date.isoformat(),
                'home_team': str(row['home_team']) if 'home_team' in row and pd.notna(row['home_team']) else 'Unknown',
                'away_team': str(row['away_team']) if 'away_team' in row and pd.notna(row['away_team']) else 'Unknown',
                'game_pk': int(row['game_pk']) if 'game_pk' in row and pd.notna(row['game_pk']) else None
            })

    # Ensure JSON-serializable output
    result = {
        'batters': [convert_numpy_types(player) for player in top_batters],
        'pitchers': [convert_numpy_types(player) for player in top_pitchers]
    }

    logging.info(f"Prediction complete. Top batters: {len(top_batters)}, Top pitchers: {len(top_pitchers)}")
    return result

# Example usage
if __name__ == "__main__":
    try:
        prediction_date = datetime.now().date()  # Or specify: datetime(2024, 10, 1).date()
        top_performers = predict_top_performers(prediction_date)
        print(json.dumps(top_performers, indent=2))
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        traceback.print_exc()
