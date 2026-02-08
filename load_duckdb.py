import duckdb
import os


BASE_PATH = r" " # Your Dir


db_path = os.path.join(BASE_PATH, "self_service_bi.duckdb")
con = duckdb.connect(db_path)

print("Connected to DuckDB.")



con.execute(f"""
CREATE OR REPLACE TABLE funds AS
SELECT * FROM read_csv_auto('{BASE_PATH}/funds.csv');
""")

con.execute(f"""
CREATE OR REPLACE TABLE fund_metrics AS
SELECT
    CAST(date AS DATE) AS date,
    *
FROM read_csv_auto('{BASE_PATH}/fund_metrics.csv');
""")

con.execute(f"""
CREATE OR REPLACE TABLE operations AS
SELECT
    CAST(date AS DATE) AS date,
    *
FROM read_csv_auto('{BASE_PATH}/operations.csv');
""")

con.execute(f"""
CREATE OR REPLACE TABLE hr_data AS
SELECT
    CAST(date AS DATE) AS date,
    *
FROM read_csv_auto('{BASE_PATH}/hr_data.csv');
""")

con.execute(f"""
CREATE OR REPLACE TABLE financials AS
SELECT
    CAST(date AS DATE) AS date,
    *
FROM read_csv_auto('{BASE_PATH}/financials.csv');
""")

print("Tables loaded.")



con.execute("""
CREATE OR REPLACE VIEW vw_fund_analytics AS
SELECT
    fm.date,
    f.fund_name,
    f.strategy,
    fm.nav,
    fm.aum,
    fm.performance_pct,
    fm.inflow_amount,
    fm.outflow_amount,
    fm.net_flow
FROM fund_metrics fm
LEFT JOIN funds f
ON fm.fund_id = f.fund_id;
""")

print("Analytics view created.")

con.close()
print("DuckDB setup completed successfully.")
