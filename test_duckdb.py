import duckdb

con = duckdb.connect(r"") # Your Dir

df = con.execute("""
SELECT fund_name, SUM(inflow_amount) AS total_inflow
FROM vw_fund_analytics
GROUP BY fund_name
ORDER BY total_inflow DESC
LIMIT 5
""").df()

print(df)
