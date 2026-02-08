import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Make sure your .env file exists in the project root."
    )

client = OpenAI(api_key=api_key)


SCHEMA = """
Tables available:

vw_fund_analytics(date, fund_name, strategy, nav, aum,
performance_pct, inflow_amount, outflow_amount, net_flow)

operations(transaction_id, date, fund_id, transaction_amount,
fee_amount, status)

hr_data(date, department, headcount, monthly_cost)

financials(date, revenue, costs, budget)
"""


SYSTEM_PROMPT = f"""
You are a BI analyst.
Your job is to translate user questions into DuckDB SQL queries.

Rules:
- ONLY return SQL.
- NEVER include markdown formatting.
- Use DuckDB syntax.
- Prefer vw_fund_analytics for fund analysis.
- Aggregate when necessary.
- Never invent columns.
- Never explain anything.
- Do not add ```sql or ```.

Schema:
{SCHEMA}
"""


def generate_sql(user_question: str):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]
    )

    sql = response.choices[0].message.content

    if not sql:
        raise ValueError("AI returned empty response.")

    
    sql = sql.replace("```sql", "")
    sql = sql.replace("```", "")
    sql = sql.strip()

    return sql
