import duckdb

db_file = 'mlb_data2.db'
output_parquet_file_stats = 'season_stats.parquet'
output_parquet_file_pitch = 'pitch_data.parquet'
table1_name = 'season_stats'
table2_name = 'pitch_data'

try:
    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute(f"ATTACH '{db_file}' (TYPE sqlite)")

    # Convert season_stats table
    con.execute(f"COPY mlb_data2.{table1_name} TO '{output_parquet_file_stats}' (FORMAT 'parquet')")
    print(f"Successfully converted table '{table1_name}' from '{db_file}' to '{output_parquet_file_stats}'")

    # Convert pitch_data table
    con.execute(f"COPY mlb_data2.{table2_name} TO '{output_parquet_file_pitch}' (FORMAT 'parquet')")
    print(f"Successfully converted table '{table2_name}' from '{db_file}' to '{output_parquet_file_pitch}'")

    con.close()

except Exception as e:
    print(f"An error occurred: {e}")
