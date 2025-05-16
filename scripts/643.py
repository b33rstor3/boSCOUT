# mlb_game_simulator.py
import statsapi  # You'll need to install this: pip install MLB-StatsAPI
import random
import pandas as pd
from datetime import datetime, date, timezone  # Added timezone
import time  # For a slight delay to make simulation feel more "live"

# Constants from the original script
BALLPARK_COORDS = {
    'ARI': {'lat': 33.4456, 'lon': -112.0667, 'name': 'Chase Field'},
    'ATL': {'lat': 33.8908, 'lon': -84.4678, 'name': 'Truist Park'},
    'BAL': {'lat': 39.2838, 'lon': -76.6217, 'name': 'Oriole Park at Camden Yards'},
    'BOS': {'lat': 42.3467, 'lon': -71.0972, 'name': 'Fenway Park'},
    'CHC': {'lat': 41.9484, 'lon': -87.6553, 'name': 'Wrigley Field'},
    'CWS': {'lat': 41.8300, 'lon': -87.6338, 'name': 'Guaranteed Rate Field'},
    'CIN': {'lat': 39.0974, 'lon': -84.5070, 'name': 'Great American Ball Park'},
    'CLE': {'lat': 41.4960, 'lon': -81.6852, 'name': 'Progressive Field'},
    'COL': {'lat': 39.7562, 'lon': -104.9942, 'name': 'Coors Field'},
    'DET': {'lat': 42.3390, 'lon': -83.0485, 'name': 'Comerica Park'},
    'HOU': {'lat': 29.7573, 'lon': -95.3555, 'name': 'Minute Maid Park'},
    'KC': {'lat': 39.0517, 'lon': -94.4803, 'name': 'Kauffman Stadium'},
    'LAA': {'lat': 33.8003, 'lon': -117.8827, 'name': 'Angel Stadium'},
    'LAD': {'lat': 34.0739, 'lon': -118.2400, 'name': 'Dodger Stadium'},
    'MIA': {'lat': 25.7781, 'lon': -80.2196, 'name': 'loanDepot park'},
    'MIL': {'lat': 43.0280, 'lon': -87.9712, 'name': 'American Family Field'},
    'MIN': {'lat': 44.9817, 'lon': -93.2777, 'name': 'Target Field'},
    'NYM': {'lat': 40.7571, 'lon': -73.8458, 'name': 'Citi Field'},
    'NYY': {'lat': 40.8296, 'lon': -73.9262, 'name': 'Yankee Stadium'},
    'OAK': {'lat': 37.7510, 'lon': -122.2009, 'name': 'Oakland Coliseum'},
    'PHI': {'lat': 39.9061, 'lon': -75.1665, 'name': 'Citizens Bank Park'},
    'PIT': {'lat': 40.4469, 'lon': -80.0057, 'name': 'PNC Park'},
    'SD': {'lat': 32.7076, 'lon': -117.1570, 'name': 'Petco Park'},
    'SEA': {'lat': 47.5914, 'lon': -122.3325, 'name': 'T-Mobile Park'},
    'SF': {'lat': 37.7786, 'lon': -122.3893, 'name': 'Oracle Park'},
    'STL': {'lat': 38.6226, 'lon': -90.1928, 'name': 'Busch Stadium'},
    'TB': {'lat': 27.7682, 'lon': -82.6534, 'name': 'Tropicana Field'},
    'TEX': {'lat': 32.7513, 'lon': -97.0828, 'name': 'Globe Life Field'},
    'TOR': {'lat': 43.6414, 'lon': -79.3894, 'name': 'Rogers Centre'},
    'WSH': {'lat': 38.8730, 'lon': -77.0074, 'name': 'Nationals Park'},
    'DEFAULT': {'lat': 39.8283, 'lon': -98.5795, 'name': 'Unknown'}
}
TEAM_ABBR_MAP = {
    'CWS': 'CWS', 'CHW': 'CWS', 'CHC': 'CHC', 'NYM': 'NYM', 'NYY': 'NYY',
    'LAA': 'LAA', 'ANA': 'LAA', 'SD': 'SD', 'SDP': 'SD', 'SF': 'SF', 'SFG': 'SF',
    'TB': 'TB', 'TBR': 'TB', 'KC': 'KC', 'KCR': 'KC', 'WSH': 'WSH', 'WSN': 'WSH',
}

# --- Player Stat Initialization ---
def init_player_stats(player_id, player_name, position):
    stats = {
        'id': player_id,
        'name': player_name,
        'pos': position,
        'PA': 0, 'AB': 0, 'R': 0, 'H': 0, '2B': 0, '3B': 0, 'HR': 0,
        'RBI': 0, 'BB': 0, 'SO': 0, 'SB': 0, 'CS': 0
    }
    if position == "P":  # Pitcher specific
        stats.update({
            'IP': 0.0, 'ER': 0, 'P_SO': 0, 'P_BB': 0, 'P_HR': 0, 'BF': 0,  # Batters Faced
            'Pitches': 0, 'Strikes': 0  # Simplified pitch counts
        })
    return stats

# --- Simplified At-Bat Simulation ---
GENERIC_PROBS = {
    'OUT_SO': 0.22,
    'OUT_OTHER': 0.41,
    'WALK': 0.10,
    'SINGLE': 0.17,
    'DOUBLE': 0.06,
    'TRIPLE': 0.005,
    'HOMERUN': 0.035
}
total_prob = sum(GENERIC_PROBS.values())
if abs(total_prob - 1.0) > 0.001:
    GENERIC_PROBS['OUT_OTHER'] += (1.0 - total_prob)

def simulate_at_bat(batter_stats, pitcher_stats):
    outcome = random.choices(list(GENERIC_PROBS.keys()), weights=list(GENERIC_PROBS.values()), k=1)[0]
    batter_stats['PA'] += 1
    pitcher_stats['BF'] += 1
    pitches_this_ab = random.randint(1, 6)
    pitcher_stats['Pitches'] += pitches_this_ab
    
    # Calculate strikes: 60-80% of pitches are strikes, with at least 1 strike unless walk with 1-2 pitches
    if outcome == 'WALK' and pitches_this_ab <= 2:
        strikes = 0  # Minimal pitches on walk may have no strikes
    else:
        strikes = max(1, int(pitches_this_ab * random.uniform(0.6, 0.8)))
        strikes = min(strikes, pitches_this_ab)  # Ensure strikes don't exceed pitches
    pitcher_stats['Strikes'] += strikes

    if outcome == 'WALK':
        batter_stats['BB'] += 1
        pitcher_stats['P_BB'] += 1
        return outcome, 0, 0, 1
    elif outcome.startswith('OUT'):
        batter_stats['AB'] += 1
        if outcome == 'OUT_SO':
            batter_stats['SO'] += 1
            pitcher_stats['P_SO'] += 1
        return outcome, 0, 1, 0
    else:  # Hit
        batter_stats['AB'] += 1
        batter_stats['H'] += 1
        pitcher_stats['Pitches'] += random.randint(0, 2)
        if outcome == 'SINGLE': return outcome, 0, 0, 1
        elif outcome == 'DOUBLE':
            batter_stats['2B'] += 1
            return outcome, 0, 0, 2
        elif outcome == 'TRIPLE':
            batter_stats['3B'] += 1
            return outcome, 0, 0, 3
        elif outcome == 'HOMERUN':
            batter_stats['HR'] += 1
            pitcher_stats['P_HR'] += 1
            return outcome, 1, 0, 4

def advance_runners_and_score(outcome, base_reached_by_batter, runners_on_base, batter_stats, pitcher_stats, all_player_stats_dict):
    runs_this_play = 0
    batter_id = batter_stats['id']
    new_runners = [None, None, None]  # 1B, 2B, 3B (stores player_id)

    # Simulate runner advancement (simplified)
    # Runner on 3rd
    if runners_on_base[2] is not None:
        runs_this_play += 1
        if runners_on_base[2] in all_player_stats_dict: all_player_stats_dict[runners_on_base[2]]['R'] += 1
    # Runner on 2nd
    if runners_on_base[1] is not None:
        if base_reached_by_batter >= 2 or outcome == 'SINGLE':  # Scores on Single or better
            runs_this_play += 1
            if runners_on_base[1] in all_player_stats_dict: all_player_stats_dict[runners_on_base[1]]['R'] += 1
        elif outcome == 'WALK' and runners_on_base[0] is not None and runners_on_base[2] is None:  # forced to 3rd by walk
            new_runners[2] = runners_on_base[1]
        elif base_reached_by_batter == 1 and new_runners[2] is None:  # single moves to 3rd
            new_runners[2] = runners_on_base[1]
    # Runner on 1st
    if runners_on_base[0] is not None:
        if base_reached_by_batter >= 3:  # Scores on Triple or HR
            runs_this_play += 1
            if runners_on_base[0] in all_player_stats_dict: all_player_stats_dict[runners_on_base[0]]['R'] += 1
        elif base_reached_by_batter == 2 and new_runners[2] is None:  # Double, runner to 3rd (simplified)
            new_runners[2] = runners_on_base[0]
        elif base_reached_by_batter == 2 and new_runners[2] is not None:  # Double, runner on 1st scores if 3rd was taken
            runs_this_play += 1
            if runners_on_base[0] in all_player_stats_dict: all_player_stats_dict[runners_on_base[0]]['R'] += 1
        elif base_reached_by_batter == 1:  # Single, runner to 2nd
            if new_runners[1] is None: new_runners[1] = runners_on_base[0]
            elif new_runners[2] is None: new_runners[2] = runners_on_base[0]  # if 2b taken, Rfrom1 to 3rd
        elif outcome == 'WALK':  # Walk, runner from 1实在是 moves to 2nd
            if new_runners[1] is None: new_runners[1] = runners_on_base[0]

    # Place batter
    if outcome == 'WALK':
        if new_runners[0] is None: new_runners[0] = batter_id
        elif new_runners[1] is None: new_runners[1] = batter_id  # Forced from 1st
        elif new_runners[2] is None: new_runners[2] = batter_id  # Forced from 2nd
    elif base_reached_by_batter == 1: new_runners[0] = batter_id
    elif base_reached_by_batter == 2: new_runners[1] = batter_id
    elif base_reached_by_batter == 3: new_runners[2] = batter_id
    elif base_reached_by_batter == 4:  # Homerun, batter scores
        runs_this_play += 1  # Batter scores themself
        batter_stats['R'] += 1

    # RBI Logic
    if outcome == 'HOMERUN':
        batter_stats['RBI'] += (1 + sum(1 for r_id in runners_on_base if r_id is not None))
    elif runs_this_play > 0 and not outcome.startswith('OUT'):
        batter_stats['RBI'] += runs_this_play

    pitcher_stats['ER'] += runs_this_play
    return runs_this_play, new_runners

all_player_stats = {}  # Global dict to hold all player stat objects

def format_statline(player_stats_id, is_pitcher=False):
    global all_player_stats
    player_stats_obj = all_player_stats.get(player_stats_id)

    if not player_stats_obj:
        print(f"Statline N/A for player ID {player_stats_id}")
        return

    name_pos = f"{player_stats_obj.get('name', f'Player {player_stats_id}')} ({player_stats_obj.get('pos', 'N/A')})"

    if is_pitcher:
        ip = player_stats_obj.get('IP', 0.0)
        ip_major = int(ip)
        ip_minor = int(round((ip - ip_major) * 3))
        print(f"{name_pos:<30} IP: {ip_major}.{ip_minor}, "
              f"R: {player_stats_obj.get('ER',0)}, ER: {player_stats_obj.get('ER',0)}, "
              f"BB: {player_stats_obj.get('P_BB',0)}, SO: {player_stats_obj.get('P_SO',0)}, HR: {player_stats_obj.get('P_HR',0)}, "
              f"PC-ST: {player_stats_obj.get('Pitches',0)}-{player_stats_obj.get('Strikes',0)}")
    else:
        print(f"{name_pos:<30} AB: {player_stats_obj.get('AB',0)}, R: {player_stats_obj.get('R',0)}, H: {player_stats_obj.get('H',0)}, "
              f"2B: {player_stats_obj.get('2B',0)}, 3B: {player_stats_obj.get('3B',0)}, HR: {player_stats_obj.get('HR',0)}, "
              f"RBI: {player_stats_obj.get('RBI',0)}, BB: {player_stats_obj.get('BB',0)}, SO: {player_stats_obj.get('SO',0)}")

def run_simulation(game_pk, home_team_name, away_team_name, home_lineup_ids, away_lineup_ids, home_sp_id, away_sp_id):
    global all_player_stats
    all_player_stats = {}

    print(f"\nSimulating {away_team_name} at {home_team_name}...")

    home_score = 0
    away_score = 0
    inning = 1
    top_of_inning = True
    game_over = False

    print("Fetching player names (this might take a moment)...")
    def get_player_details(player_id, default_pos_prefix=""):
        if not player_id: return None
        try:
            player_info = statsapi.get('player', {'personId': player_id})
            if player_info and player_info.get('people'):
                name = player_info['people'][0].get('fullName', f"Player {player_id}")
                primary_pos_data = player_info['people'][0].get('primaryPosition', {})
                pos_code = primary_pos_data.get('abbreviation', default_pos_prefix)
            else:
                name = f"Player {player_id}"
                pos_code = default_pos_prefix

            if player_id == home_sp_id or player_id == away_sp_id: pos_code = "P"
            elif not pos_code and default_pos_prefix: pos_code = default_pos_prefix
            elif not pos_code: pos_code = "POS"

            stats = init_player_stats(player_id, name, pos_code)
            all_player_stats[player_id] = stats
            return stats
        except Exception as e:
            name = f"Player {player_id}"
            pos_code = "P" if player_id in [home_sp_id, away_sp_id] else default_pos_prefix if default_pos_prefix else "N/A"
            stats = init_player_stats(player_id, name, pos_code)
            all_player_stats[player_id] = stats
            return stats

    home_sp_stats_obj = get_player_details(home_sp_id, "P")
    away_sp_stats_obj = get_player_details(away_sp_id, "P")

    home_lineup_stats_objs = [get_player_details(pid, f"H{i+1}") for i, pid in enumerate(home_lineup_ids)]
    away_lineup_stats_objs = [get_player_details(pid, f"V{i+1}") for i, pid in enumerate(away_lineup_ids)]

    home_lineup_stats_objs = [p for p in home_lineup_stats_objs if p]
    away_lineup_stats_objs = [p for p in away_lineup_stats_objs if p]

    if not home_sp_stats_obj or not away_sp_stats_obj or not home_lineup_stats_objs or not away_lineup_stats_objs:
        print("Critical error: Could not initialize starting players/pitchers. Aborting simulation.")
        return

    home_batter_idx = 0
    away_batter_idx = 0

    print("\n--- Starting Simulation ---")
    time.sleep(0.5)

    while not game_over:
        print("-" * 40)
        inning_half_str = "Top" if top_of_inning else "Bottom"
        print(f"{inning_half_str} of Inning {inning}   |   {away_team_name}: {away_score} - {home_team_name}: {home_score}")

        outs = 0
        runners_on_base = [None, None, None]

        current_lineup_stats = away_lineup_stats_objs if top_of_inning else home_lineup_stats_objs
        current_pitcher_stats_obj = home_sp_stats_obj if top_of_inning else away_sp_stats_obj

        if not current_lineup_stats:
            print(f"Error: No lineup for {'Away' if top_of_inning else 'Home'}. Ending inning.")
            outs = 3

        while outs < 3:
            current_batter_idx = away_batter_idx if top_of_inning else home_batter_idx
            if current_batter_idx >= len(current_lineup_stats):
                outs = 3
                break

            batter_stats_obj = current_lineup_stats[current_batter_idx]

            print(f"\nNow Batting: {batter_stats_obj['name']} ({batter_stats_obj['pos']}) for {'Away' if top_of_inning else 'Home'}")
            time.sleep(0.05)

            outcome, runs_on_play_direct_hr, outs_on_play, base_reached = simulate_at_bat(batter_stats_obj, current_pitcher_stats_obj)
            print(f"Outcome: {outcome}")

            if outs_on_play > 0:
                outs += outs_on_play
                current_pitcher_stats_obj['IP'] += (outs_on_play / 3.0)
                if outs >= 3:
                    print(f"Three outs. Inning over.")
                    break
            else:
                runs_scored_this_ab, new_runners = advance_runners_and_score(
                    outcome, base_reached, runners_on_base, batter_stats_obj, current_pitcher_stats_obj, all_player_stats
                )
                runners_on_base = new_runners

                if outcome == 'HOMERUN':
                    hr_rbi = sum(1 for r_id in runners_on_base if r_id is not None and r_id != batter_stats_obj['id']) + 1
                    print(f"HOMERUN! {batter_stats_obj['name']} scores {hr_rbi - 1} runner(s) and themself!")
                elif runs_scored_this_ab > 0:
                    print(f"{batter_stats_obj['name']} drives in {runs_scored_this_ab} run(s)!")

                if top_of_inning: away_score += runs_scored_this_ab
                else: home_score += runs_scored_this_ab

            if outs_on_play == 0 and outs < 3:
                runner_names = [all_player_stats.get(r_id, {}).get('name', 'Empty') if r_id else '-' for r_id in runners_on_base]
                print(f"Runners on: 1B:{runner_names[0]}, 2B:{runner_names[1]}, 3B:{runner_names[2]}")

            if top_of_inning:
                away_batter_idx = (away_batter_idx + 1) % len(away_lineup_stats_objs)
            else:
                home_batter_idx = (home_batter_idx + 1) % len(home_lineup_stats_objs)

            print(f"Score: {away_team_name}: {away_score} - {home_team_name}: {home_score}, Outs: {outs}")
            time.sleep(0.1)

        # End of half-inning logic
        if top_of_inning:
            top_of_inning = False
            if inning >= 9 and home_score > away_score:
                game_over = True
        else:
            top_of_inning = True
            if inning >= 9 and home_score != away_score:
                game_over = True
            elif inning >= 9 and home_score == away_score:
                inning += 1
            elif inning < 9:
                inning += 1

        if inning > 15:
            print("Game reached 15 innings, calling it.")
            game_over = True

    print("\n" + "="*20 + " FINAL " + "="*20)
    print(f"{away_team_name}: {away_score}")
    print(f"{home_team_name}: {home_score}")
    winner = "Tie"
    if away_score > home_score: winner = away_team_name
    elif home_score > away_score: winner = home_team_name
    print(f"Winner: {winner}")

    print("\n" + "--- Predicted Statlines ---")
    print(f"\n{away_team_name.upper()} Batters:")
    for player_id in away_lineup_ids: format_statline(player_id)
    print(f"\n{away_team_name.upper()} Pitcher:")
    if away_sp_id: format_statline(away_sp_id, is_pitcher=True)

    print(f"\n{home_team_name.upper()} Batters:")
    for player_id in home_lineup_ids: format_statline(player_id)
    print(f"\n{home_team_name.upper()} Pitcher:")
    if home_sp_id: format_statline(home_sp_id, is_pitcher=True)

def get_generic_lineup(team_abbr, is_home):
    """Generate a generic lineup with placeholder player IDs and names."""
    lineup = []
    prefix = "H" if is_home else "V"
    for i in range(9):
        player_id = f"generic_{team_abbr}_{prefix}{i+1}"
        player_name = f"{team_abbr} Player {i+1}"
        position = f"{prefix}{i+1}"
        lineup.append((player_id, player_name, position))
    return lineup

def get_game_details(game):
    """Extract game details and handle missing data."""
    game_pk = game.get('game_pk', None)
    away_team = game.get('away_name', 'Unknown Away Team')
    home_team = game.get('home_name', 'Unknown Home Team')
    away_id = game.get('away_id', None)
    home_id = game.get('home_id', None)
    status = game.get('status', 'Scheduled')
    game_date = game.get('game_date', datetime.now().strftime('%Y-%m-%d'))

    # Fetch lineups and starting pitchers if game_pk is available and game is not in future
    away_lineup_ids = []
    home_lineup_ids = []
    away_sp_id = None
    home_sp_id = None

    if game_pk and status in ['Scheduled', 'In Progress', 'Final']:
        try:
            game_info = statsapi.boxscore_data(game_pk)
            away_lineup_ids = game_info.get('away', {}).get('battingOrder', [])
            home_lineup_ids = game_info.get('home', {}).get('battingOrder', [])
            away_sp_id = game_info.get('away', {}).get('pitchers', [None])[0]
            home_sp_id = game_info.get('home', {}).get('pitchers', [None])[0]
        except Exception as e:
            print(f"Warning: Could not fetch boxscore data for game_pk {game_pk}: {e}")

    # Fallback to generic lineups if data is missing
    if not away_lineup_ids:
        away_lineup_ids = [pid for pid, _, _ in get_generic_lineup(away_team[:3].upper(), is_home=False)]
    if not home_lineup_ids:
        home_lineup_ids = [pid for pid, _, _ in get_generic_lineup(home_team[:3].upper(), is_home=True)]
    if not away_sp_id:
        away_sp_id = f"generic_{away_team[:3].upper()}_P"
    if not home_sp_id:
        home_sp_id = f"generic_{home_team[:3].upper()}_P"

    return {
        'game_pk': game_pk,
        'away_team': away_team,
        'home_team': home_team,
        'away_id': away_id,
        'home_id': home_id,
        'away_lineup_ids': away_lineup_ids,
        'home_lineup_ids': home_lineup_ids,
        'away_sp_id': away_sp_id,
        'home_sp_id': home_sp_id,
        'status': status,
        'game_date': game_date
    }

def main():
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching games for today: {today_str}")
    try:
        games_today = statsapi.schedule(date=today_str)
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        print("This could be due to no games scheduled, or an API/network issue.")
        return

    if not games_today:
        print("No games scheduled for today or unable to fetch games.")
        return

    print("\nToday's Games:")
    for i, game in enumerate(games_today):
        game_dt_str = game.get('game_datetime', '')
        display_time_str = 'Time N/A'
        game_date_str = game.get('game_date', 'Date N/A')

        if game_dt_str:
            try:
                game_dt_utc = datetime.strptime(game_dt_str, '%Y-%m-%dT%H:%M:%SZ')
                game_dt_utc = game_dt_utc.replace(tzinfo=timezone.utc)
                game_dt_local = game_dt_utc.astimezone(tz=None)
                display_time_str = game_dt_local.strftime('%I:%M %p %Z')
            except ValueError:
                display_time_str = game_dt_str.split('T')[-1].replace('Z', ' UTC') if 'T' in game_dt_str else game_dt_str

        away_name = game.get('away_name', 'Away Team N/A')
        home_name = game.get('home_name', 'Home Team N/A')
        status = game.get('status', 'Status N/A')
        print(f"{i+1}. {away_name} at {home_name} ({game_date_str} {display_time_str}) - Status: {status}")

    while True:
        try:
            choice_input = input("Select a game to simulate (number, or 'q' to quit): ")
            if choice_input.lower() == 'q':
                print("Exiting simulator.")
                return
            choice = int(choice_input) - 1
            if 0 <= choice < len(games_today):
                selected_game = games_today[choice]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Extract game details
    game_details = get_game_details(selected_game)
    game_pk = game_details['game_pk']
    away_team = game_details['away_team']
    home_team = game_details['home_team']
    away_lineup_ids = game_details['away_lineup_ids']
    home_lineup_ids = game_details['home_lineup_ids']
    away_sp_id = game_details['away_sp_id']
    home_sp_id = game_details['home_sp_id']
    status = game_details['status']

    print(f"\nSelected Game: {away_team} at {home_team} - Status: {status}")
    if status == 'In Progress':
        print("Note: Game is in progress. Simulation will predict outcome based on generic or available data.")
    elif status == 'Final':
        print("Note: Game has ended. Simulation will predict a hypothetical outcome.")
    elif not game_pk:
        print("Warning: Game ID missing. Using generic simulation data.")

    # Run the simulation
    run_simulation(
        game_pk=game_pk,
        home_team_name=home_team,
        away_team_name=away_team,
        home_lineup_ids=home_lineup_ids,
        away_lineup_ids=away_lineup_ids,
        home_sp_id=home_sp_id,
        away_sp_id=away_sp_id
    )

if __name__ == "__main__":
    main()
