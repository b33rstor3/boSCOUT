import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Load the merged dataframe
df = pd.read_csv('lahman_data_processed/merged_lahman_2010_2023.csv', low_memory=False)

# Enrich with SQLite data
def enrich_with_sqlite_data(lahman_df, db_path='mlb_data2.db', table_name='statcast'):
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
        SELECT playerID, yearID,
               AVG(pitch_velocity) as avg_pitch_velocity,
               AVG(exit_velocity) as avg_exit_velocity
        FROM {table_name}
        WHERE yearID BETWEEN 2010 AND 2023
        GROUP BY playerID, yearID
        """
        sqlite_df = pd.read_sql_query(query, conn)
        conn.close()
        enriched_df = lahman_df.merge(sqlite_df, on=['playerID', 'yearID'], how='left')
        enriched_df['avg_pitch_velocity'] = enriched_df['avg_pitch_velocity'].fillna(enriched_df['avg_pitch_velocity'].median())
        enriched_df['avg_exit_velocity'] = enriched_df['avg_exit_velocity'].fillna(enriched_df['avg_exit_velocity'].median())
        print("\nEnriched dataset with SQLite features:")
        print(enriched_df[['playerID', 'yearID', 'avg_pitch_velocity', 'avg_exit_velocity']].head())
        return enriched_df
    except Exception as e:
        print(f"Error enriching with SQLite data: {e}")
        return lahman_df

# Plot correlation matrix
def plot_correlation_matrix(df, features, target=None, filename='correlation_matrix.png'):
    corr = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title(f'Correlation Matrix (Target: {target if target else "None"})')
    plt.savefig(filename)
    plt.close()
    if target:
        target_corr = corr[target].sort_values(ascending=False)
        print(f"\nCorrelations with {target}:")
        print(target_corr)

# Compute feature importance
def compute_feature_importance(df, features, target, filename='feature_importance.png'):
    X = df[features].fillna(0)
    y = df[target].fillna(0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\nFeature ranking for {target}:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {features[indices[f]]} ({importances[indices[f]]:.4f})")
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances for {target}")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate interaction features and mutual information
def generate_interaction_features(df, features, target, prefix=''):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X = df[features].fillna(0)
    poly_features = poly.fit_transform(X)
    poly_feature_names = [f"{prefix}{name}" for name in poly.get_feature_names_out(features)]
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df_combined = pd.concat([df, poly_df], axis=1)
    mi_scores = mutual_info_regression(poly_features, df[target].fillna(0))
    mi_df = pd.DataFrame({'Feature': poly_feature_names, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values('MI_Score', ascending=False)
    print(f"\nMutual Information Scores for {target} (Top 10):")
    print(mi_df.head(10))
    return df_combined, mi_df

# Compute rolling stats and breakout detection
def compute_rolling_stats_and_breakouts(df, target, group_col='playerID', time_col='yearID', window=2, threshold=1.5):
    df_sorted = df.sort_values([group_col, time_col])
    df_sorted[f'{target}_roll_mean'] = df_sorted.groupby(group_col)[target].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_sorted[f'{target}_roll_std'] = df_sorted.groupby(group_col)[target].transform(lambda x: x.rolling(window, min_periods=1).std())
    df_sorted[f'{target}_breakout'] = (df_sorted[target] > df_sorted[f'{target}_roll_mean'] + threshold * df_sorted[f'{target}_roll_std']).astype(int)
    breakout_players = df_sorted[df_sorted[f'{target}_breakout'] == 1][group_col].unique()[:5]
    for player in breakout_players:
        player_data = df_sorted[df_sorted[group_col] == player]
        plt.figure(figsize=(10, 6))
        plt.plot(player_data[time_col], player_data[target], label=target)
        plt.plot(player_data[time_col], player_data[f'{target}_roll_mean'], label='Rolling Mean')
        plt.scatter(player_data[player_data[f'{target}_breakout'] == 1][time_col], 
                    player_data[player_data[f'{target}_breakout'] == 1][target], c='red', label='Breakout')
        plt.title(f"{target} Trend for {player}")
        plt.legend()
        plt.savefig(f'lahman_data_processed/{target}_trend_{player}.png')
        plt.close()
    print(f"\nNumber of breakout seasons for {target}: {df_sorted[f'{target}_breakout'].sum()}")
    return df_sorted

# Compute lag correlation
def compute_lag_correlation(df, target, group_col='playerID', time_col='yearID'):
    df_sorted = df.sort_values([group_col, time_col])
    df_sorted[f'{target}_lag1'] = df_sorted.groupby(group_col)[target].shift(1)
    lag_corr = df_sorted[[target, f'{target}_lag1']].corr().iloc[0, 1]
    print(f"\nCorrelation between {target} and {target}_lag1: {lag_corr:.4f}")

# Enrich data
print("Enriching Lahman data with SQLite game-level metrics...")
df_enriched = enrich_with_sqlite_data(df)
df_enriched.to_csv('lahman_data_processed/enriched_lahman_2010_2023.csv', index=False)

# Separate into batters and pitchers
batters = df_enriched[df_enriched['AB'] > 0].copy()
pitchers = df_enriched[df_enriched['IPouts'] > 0].copy()

# Analysis for batters
print("\n=== Analyzing Batters ===")
target_batters = 'HR'
features_batters = [col for col in batters.columns 
                    if batters[col].dtype in ['float64', 'int64'] and col != target_batters 
                    and col not in ['ID', 'yearID']] + ['avg_exit_velocity']
plot_correlation_matrix(batters, features_batters, target_batters, 'correlation_matrix_batters.png')
compute_feature_importance(batters, features_batters, target_batters, 'feature_importance_batters.png')
compute_lag_correlation(batters, target_batters)
print("Generating interaction features for batters...")
batters_combined, mi_batters = generate_interaction_features(batters, features_batters, target_batters, 'bat_')
batters_combined.to_csv('lahman_data_processed/batters_engineered_features.csv', index=False)
mi_batters.to_csv('lahman_data_processed/batters_mi_scores.csv', index=False)
print("Computing rolling stats and breakouts for batters...")
batters_rolling = compute_rolling_stats_and_breakouts(batters, target_batters)
batters_rolling.to_csv('lahman_data_processed/batters_rolling_breakouts.csv', index=False)

# Analysis for pitchers
print("\n=== Analyzing Pitchers ===")
target_pitchers = 'SO_pitching'
features_pitchers = [col for col in pitchers.columns 
                     if pitchers[col].dtype in ['float64', 'int64'] and col != target_pitchers 
                     and col not in ['ID', 'yearID']] + ['avg_pitch_velocity']
plot_correlation_matrix(pitchers, features_pitchers, target_pitchers, 'correlation_matrix_pitchers.png')
compute_feature_importance(pitchers, features_pitchers, target_pitchers, 'feature_importance_pitchers.png')
compute_lag_correlation(pitchers, target_pitchers)
print("Generating interaction features for pitchers...")
pitchers_combined, mi_pitchers = generate_interaction_features(pitchers, features_pitchers, target_pitchers, 'pit_')
pitchers_combined.to_csv('lahman_data_processed/pitchers_engineered_features.csv', index=False)
mi_pitchers.to_csv('lahman_data_processed/pitchers_mi_scores.csv', index=False)
print("Computing rolling stats and breakouts for pitchers...")
pitchers_rolling = compute_rolling_stats_and_breakouts(pitchers, target_pitchers)
pitchers_rolling.to_csv('lahman_data_processed/pitchers_rolling_breakouts.csv', index=False)

print("\nAnalysis complete. Check saved plots and CSVs for results.")
