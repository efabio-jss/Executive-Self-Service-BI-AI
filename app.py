import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from ai.ai_query_engine import generate_sql


DB_PATH = r"" # Your Dir
st.set_page_config(page_title="Executive AI BI", layout="wide")


st.markdown(
    """
<style>
:root{
  --bg:#231E35;
  --panel:#2E2847;
  --panel2:#332C52;
  --stroke:rgba(255,255,255,0.09);
  --text:#EAEAF2;
  --muted:rgba(234,234,242,0.70);
  --good:#3EE6A2;
  --bad:#FF5C7A;
  --accent:#5DA9FF;
  --accent2:#7B6CFF;
}

html, body, [class*="css"] { background: var(--bg) !important; color: var(--text) !important; }

/* FIX: evita header "cortado" por falta de espaço no topo */
.block-container { padding-top: 3.0rem; padding-bottom: 2.0rem; max-width: 1600px; }

.hdr{
  margin-top: 0.25rem;
  background: linear-gradient(135deg, rgba(93,169,255,0.18), rgba(123,108,255,0.12));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 16px;
}
.hdr-title{ font-size: 1.35rem; font-weight: 800; letter-spacing: 0.2px; }
.hdr-sub{ color: var(--muted); font-size: 0.92rem; }

.panel{
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 14px;
}

.panel-title{
  font-weight: 750;
  margin-bottom: 6px;
}

.kpi{
  background: var(--panel2);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 12px 12px;
}
.kpi-label{ color: var(--muted); font-size: 0.86rem; }
.kpi-value{ font-size: 1.55rem; font-weight: 900; margin-top: 2px; }
.kpi-delta{ font-size: 0.90rem; font-weight: 700; margin-top: 4px; }
.kpi-good{ color: var(--good); }
.kpi-bad{ color: var(--bad); }
.kpi-muted{ color: var(--muted); }

div[data-testid="stDataFrame"]{
  background: var(--panel);
  border-radius: 18px;
  border: 1px solid var(--stroke);
  overflow: hidden;
}

hr { border: none; border-top: 1px solid var(--stroke); margin: 0.7rem 0; }

section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(46,40,71,1), rgba(35,30,53,1)) !important;
  border-right: 1px solid var(--stroke);
}

.smallnote{ color: var(--muted); font-size: 0.86rem; }
</style>
""",
    unsafe_allow_html=True
)


@st.cache_resource
def get_con():
    return duckdb.connect(DB_PATH, read_only=True)

con = get_con()


def fmt_eur(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e9: return f"{sign}€ {x/1e9:.2f}B"
    if x >= 1e6: return f"{sign}€ {x/1e6:.2f}M"
    if x >= 1e3: return f"{sign}€ {x/1e3:.2f}K"
    return f"{sign}€ {x:,.0f}"

def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x)*100:.2f}%"

def sql_in_list(values):
    vals = [str(v).replace("'", "''") for v in values]
    return "(" + ",".join([f"'{v}'" for v in vals]) + ")"

@st.cache_data
def get_bounds():
    dmin, dmax = con.execute("SELECT MIN(date), MAX(date) FROM fund_metrics").fetchone()
    return pd.to_datetime(dmin).date(), pd.to_datetime(dmax).date()

@st.cache_data
def get_filters():
    strategies = con.execute("SELECT DISTINCT strategy FROM funds ORDER BY 1").df()["strategy"].tolist()
    fund_names = con.execute("SELECT DISTINCT fund_name FROM funds ORDER BY 1").df()["fund_name"].tolist()
    return strategies, fund_names

def build_where(date_from, date_to, strategies, fund_names):
    clauses = []
    if date_from and date_to:
        clauses.append(f"date BETWEEN '{date_from}' AND '{date_to}'")
    if strategies:
        clauses.append(f"strategy IN {sql_in_list(strategies)}")
    if fund_names:
        clauses.append(f"fund_name IN {sql_in_list(fund_names)}")
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""

def prev_period(date_from: date, date_to: date):
    if not date_from or not date_to:
        return None, None
    delta = (date_to - date_from).days + 1
    prev_to = date_from - pd.Timedelta(days=1)
    prev_from = prev_to - pd.Timedelta(days=delta - 1)
    return prev_from, prev_to

def kpi_card(label, value, delta=None, good_when_positive=True):
    if delta is None or pd.isna(delta):
        delta_html = '<div class="kpi-delta kpi-muted">vs prev: —</div>'
    else:
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
        is_good = (delta >= 0) if good_when_positive else (delta <= 0)
        cls = "kpi-good" if is_good else "kpi-bad"
        delta_html = f'<div class="kpi-delta {cls}">vs prev: {arrow} {fmt_eur(delta) if "€" in str(value) else delta}</div>'
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )


min_d, max_d = get_bounds()
all_strategies, all_funds = get_filters()

st.sidebar.markdown("### Slicers")
date_from, date_to = st.sidebar.date_input(
    "Period",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

selected_strategies = st.sidebar.multiselect(
    "Strategy",
    options=all_strategies,
    default=all_strategies
)

selected_funds = st.sidebar.multiselect(
    "Funds (optional)",
    options=all_funds,
    default=[]
)

st.sidebar.markdown("---")
show_debug = st.sidebar.toggle("Show SQL panels", value=False)
st.sidebar.caption("Tip: keep slicers broad for C-level view.")

if st.sidebar.button("Reset slicers"):
    st.rerun()

where = build_where(date_from, date_to, selected_strategies, selected_funds)
prev_from, prev_to = prev_period(date_from, date_to)
prev_where = build_where(prev_from, prev_to, selected_strategies, selected_funds) if prev_from and prev_to else ""


st.markdown(
    f"""
    <div class="hdr">
      <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px;">
        <div>
          <div class="hdr-title">Executive Overview — AI Self-Service BI</div>
          <div class="hdr-sub">{date_from:%d %b %Y} → {date_to:%d %b %Y} · Strategies: {len(selected_strategies)} · Funds: {"All" if len(selected_funds)==0 else len(selected_funds)}</div>
        </div>
        <div class="hdr-sub">Last refresh: {max_d:%d %b %Y}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")


kpi_sql = f"""
SELECT
  SUM(aum) AS total_aum,
  SUM(net_flow) AS net_flow,
  AVG(performance_pct) AS avg_perf
FROM vw_fund_analytics
{where};
"""
total_aum, net_flow, avg_perf = con.execute(kpi_sql).fetchone()

fin_sql = f"""
SELECT
  SUM(revenue) AS revenue,
  SUM(costs) AS costs,
  SUM(budget) AS budget
FROM financials
WHERE date BETWEEN '{date_from}' AND '{date_to}';
"""
revenue, costs, budget = con.execute(fin_sql).fetchone()
profit = (revenue - costs) if revenue is not None and costs is not None else None
budget_var = (profit - budget) if profit is not None and budget is not None else None

ops_sql = f"""
SELECT
  COUNT(*) AS tx,
  SUM(CASE WHEN status='Completed' THEN 1 ELSE 0 END) AS completed,
  SUM(CASE WHEN status='Failed' THEN 1 ELSE 0 END) AS failed,
  SUM(fee_amount) AS fees
FROM operations
WHERE date BETWEEN '{date_from}' AND '{date_to}';
"""
tx, completed, failed, fees = con.execute(ops_sql).fetchone()
success_rate = (completed / tx) if tx else None

hr_sql = f"""
SELECT
  AVG(headcount) AS avg_headcount,
  SUM(monthly_cost) AS hr_cost
FROM hr_data
WHERE date BETWEEN '{date_from}' AND '{date_to}';
"""
avg_headcount, hr_cost = con.execute(hr_sql).fetchone()

if prev_where:
    kpi_prev_sql = f"""
    SELECT
      SUM(aum) AS total_aum,
      SUM(net_flow) AS net_flow,
      AVG(performance_pct) AS avg_perf
    FROM vw_fund_analytics
    {prev_where};
    """
    p_aum, p_net, p_perf = con.execute(kpi_prev_sql).fetchone()

    fin_prev_sql = f"""
    SELECT
      SUM(revenue) AS revenue,
      SUM(costs) AS costs,
      SUM(budget) AS budget
    FROM financials
    WHERE date BETWEEN '{prev_from}' AND '{prev_to}';
    """
    p_rev, p_costs, p_budget = con.execute(fin_prev_sql).fetchone()
    p_profit = (p_rev - p_costs) if p_rev is not None and p_costs is not None else None
    p_budget_var = (p_profit - p_budget) if p_profit is not None and p_budget is not None else None
else:
    p_aum = p_net = p_perf = None
    p_rev = p_costs = p_budget = None
    p_profit = p_budget_var = None

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    kpi_card("Total AUM", fmt_eur(total_aum), None if p_aum is None else (total_aum - p_aum), good_when_positive=True)
with k2:
    kpi_card("Net Flow", fmt_eur(net_flow), None if p_net is None else (net_flow - p_net), good_when_positive=True)
with k3:
    if p_perf is None or avg_perf is None:
        delta_html = None
    else:
        delta_html = (avg_perf - p_perf) * 100  
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-label">Avg Performance</div>
          <div class="kpi-value">{fmt_pct(avg_perf)}</div>
          <div class="kpi-delta {'kpi-good' if (delta_html is not None and delta_html>=0) else 'kpi-bad' if delta_html is not None else 'kpi-muted'}">
            vs prev: {"▲" if (delta_html is not None and delta_html>0) else "▼" if (delta_html is not None and delta_html<0) else "•" if delta_html==0 else ""} {"—" if delta_html is None else f"{delta_html:.2f}pp"}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with k4:
    kpi_card("Revenue", fmt_eur(revenue), None if p_rev is None else (revenue - p_rev), good_when_positive=True)
with k5:
    kpi_card("Costs", fmt_eur(costs), None if p_costs is None else (costs - p_costs), good_when_positive=False)
with k6:
    kpi_card("Budget Variance", fmt_eur(budget_var), None if p_budget_var is None else (budget_var - p_budget_var), good_when_positive=True)

st.write("")
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.markdown(f"<div class='panel'><div class='panel-title'>Operations</div><div class='smallnote'>Transactions: <b>{int(tx):,}</b> · Fees: <b>{fmt_eur(fees)}</b></div></div>", unsafe_allow_html=True)
with b2:
    sr = "—" if success_rate is None else f"{success_rate*100:.2f}%"
    st.markdown(f"<div class='panel'><div class='panel-title'>Ops Success Rate</div><div class='smallnote'><b>{sr}</b></div></div>", unsafe_allow_html=True)
with b3:
    st.markdown(f"<div class='panel'><div class='panel-title'>People</div><div class='smallnote'>Avg headcount: <b>{avg_headcount:.1f}</b></div></div>", unsafe_allow_html=True)
with b4:
    st.markdown(f"<div class='panel'><div class='panel-title'>HR Cost</div><div class='smallnote'><b>{fmt_eur(hr_cost)}</b></div></div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


row1_left, row1_right = st.columns((2, 1))

with row1_left:
    st.markdown("<div class='panel'><div class='panel-title'>AUM Trend</div></div>", unsafe_allow_html=True)
    aum_trend_sql = f"""
    SELECT date, SUM(aum) AS aum, AVG(performance_pct) AS avg_perf
    FROM vw_fund_analytics
    {where}
    GROUP BY date
    ORDER BY date;
    """
    aum_df = con.execute(aum_trend_sql).df()
    fig = px.area(aum_df, x="date", y="aum", template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=10),
        height=300,
        font=dict(color="#EAEAF2"),
    )
    fig.update_traces(hovertemplate="Month=%{x|%b %Y}<br>AUM=%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='panel'><div class='panel-title'>Net Flow Trend</div></div>", unsafe_allow_html=True)
    flow_trend_sql = f"""
    SELECT date,
           SUM(inflow_amount) AS inflow,
           SUM(outflow_amount) AS outflow,
           SUM(net_flow) AS net_flow
    FROM vw_fund_analytics
    {where}
    GROUP BY date
    ORDER BY date;
    """
    flow_df = con.execute(flow_trend_sql).df()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=flow_df["date"], y=flow_df["net_flow"], name="Net Flow"))
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=10),
        height=300,
        font=dict(color="#EAEAF2"),
        showlegend=False
    )
    fig2.update_traces(hovertemplate="Month=%{x|%b %Y}<br>Net Flow=%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig2, use_container_width=True)


with row1_right:
    st.markdown("<div class='panel'><div class='panel-title'>Top Funds (Net Flow)</div></div>", unsafe_allow_html=True)
    top_sql = f"""
    SELECT
      fund_name,
      SUM(net_flow) AS net_flow,
      SUM(inflow_amount) AS inflow,
      SUM(outflow_amount) AS outflow,
      AVG(performance_pct) AS avg_perf
    FROM vw_fund_analytics
    {where}
    GROUP BY fund_name
    ORDER BY net_flow DESC
    LIMIT 10;
    """
    top_df = con.execute(top_sql).df()
    top_df_disp = top_df.copy()
    top_df_disp["net_flow"] = top_df_disp["net_flow"].map(fmt_eur)
    top_df_disp["inflow"] = top_df_disp["inflow"].map(fmt_eur)
    top_df_disp["outflow"] = top_df_disp["outflow"].map(fmt_eur)
    top_df_disp["avg_perf"] = top_df_disp["avg_perf"].map(fmt_pct)
    st.dataframe(top_df_disp, use_container_width=True, height=260)

    st.markdown(
        """
        <div class='panel'>
            <div class='panel-title'>Premium Map — Strategy Footprint</div>
            <div class='smallnote'>Demo geo layer (synthetic hubs by strategy) for executive storytelling.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    map_sql = f"""
    SELECT strategy, SUM(aum) AS aum, SUM(net_flow) AS net_flow
    FROM vw_fund_analytics
    {where}
    GROUP BY strategy
    ORDER BY aum DESC;
    """
    map_df = con.execute(map_sql).df()

    if map_df.empty:
        st.info("No data available for the current slicers.")
    else:
        hubs = {
            "Equity": ("Madrid", 40.4168, -3.7038),
            "Fixed Income": ("Paris", 48.8566, 2.3522),
            "Infrastructure": ("Tokyo", 35.6762, 139.6503),
            "ESG": ("Berlin", 52.5200, 13.4050),
            "Multi Asset": ("London", 51.5072, -0.1276),
            "Private Debt": ("San Francisco", 37.7749, -122.4194),
            "Quant": ("Dubai", 25.2048, 55.2708),
            "Real Estate": ("Singapore", 1.3521, 103.8198),
        }

        map_df["hub"] = map_df["strategy"].map(lambda s: hubs.get(s, ("N/A", 0.0, 0.0))[0])
        map_df["lat"] = map_df["strategy"].map(lambda s: hubs.get(s, ("N/A", 0.0, 0.0))[1])
        map_df["lon"] = map_df["strategy"].map(lambda s: hubs.get(s, ("N/A", 0.0, 0.0))[2])

        max_aum = float(map_df["aum"].max()) if float(map_df["aum"].max()) != 0 else 1.0
        map_df["_size"] = 18 + 22 * (map_df["aum"] / max_aum)   
        map_df["_glow"] = map_df["_size"] * 1.55

        figm = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            color="strategy",
            size="_size",
            size_max=40,
            hover_name="hub",
            hover_data={
                "strategy": True,
                "aum": ":,.0f",
                "net_flow": ":,.0f",
                "lat": False,
                "lon": False,
                "_size": False,
                "_glow": False,
            },
            zoom=0.7,
            height=360
        )

        figm.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAEAF2"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )

        glow_trace = go.Scattermapbox(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=map_df["_glow"],
                color="rgba(255,255,255,0.18)"
            ),
            hoverinfo="skip",
            showlegend=False
        )
        figm.add_trace(glow_trace)
        figm.data = (figm.data[-1],) + figm.data[:-1]  

           
    figm.update_traces(
        marker=dict(
            opacity=0.9,
            sizemode="area"
        )
    )

    st.plotly_chart(
        figm,
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "Executive_Strategy_Map",
                "height": 800,
                "width": 1600,
                "scale": 2
            }
        }
    )




st.markdown("<hr/>", unsafe_allow_html=True)


r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    st.markdown("<div class='panel'><div class='panel-title'>Operations Status Mix</div></div>", unsafe_allow_html=True)
    ops_mix_sql = f"""
    SELECT status, COUNT(*) AS cnt, SUM(fee_amount) AS fees
    FROM operations
    WHERE date BETWEEN '{date_from}' AND '{date_to}'
    GROUP BY status
    ORDER BY cnt DESC;
    """
    ops_mix = con.execute(ops_mix_sql).df()
    fig = px.pie(ops_mix, names="status", values="cnt", hole=0.62, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), height=260, showlegend=True, font=dict(color="#EAEAF2"))
    fig.update_traces(hovertemplate="Status=%{label}<br>Count=%{value}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    st.markdown("<div class='panel'><div class='panel-title'>HR Cost by Department</div></div>", unsafe_allow_html=True)
    hr_dep_sql = f"""
    SELECT department, SUM(monthly_cost) AS cost, AVG(headcount) AS avg_headcount
    FROM hr_data
    WHERE date BETWEEN '{date_from}' AND '{date_to}'
    GROUP BY department
    ORDER BY cost DESC;
    """
    hr_dep = con.execute(hr_dep_sql).df()
    fig = px.bar(hr_dep, x="department", y="cost", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), height=260, font=dict(color="#EAEAF2"))
    fig.update_traces(hovertemplate="Dept=%{x}<br>Cost=%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

with r2c3:
    st.markdown("<div class='panel'><div class='panel-title'>Finance — Profit & Budget Variance</div></div>", unsafe_allow_html=True)
    fin_trend_sql = f"""
    SELECT date,
           revenue,
           costs,
           (revenue - costs) AS profit,
           budget,
           ((revenue - costs) - budget) AS budget_variance
    FROM financials
    WHERE date BETWEEN '{date_from}' AND '{date_to}'
    ORDER BY date;
    """
    fin = con.execute(fin_trend_sql).df()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fin["date"], y=fin["profit"], mode="lines", name="Profit"))
    fig.add_trace(go.Bar(x=fin["date"], y=fin["budget_variance"], name="Budget Var", opacity=0.75))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), height=260, font=dict(color="#EAEAF2"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_traces(hovertemplate="Month=%{x|%b %Y}<br>Value=%{y:,.0f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## AI Self-Service (Executive)")

st.markdown(
    "<div class='smallnote'>Ask questions about Funds, Operations, HR or Finance. The AI generates DuckDB SQL, executes it, and auto-visualises when possible.</div>",
    unsafe_allow_html=True
)

q = st.text_input("Ask your data (e.g., 'Top 10 funds by inflow last quarter', 'Monthly fees trend', 'HR cost by department')")

if q:
    try:
        sql = generate_sql(q)

        if show_debug:
            st.caption("Generated SQL")
            st.code(sql, language="sql")

        res = con.execute(sql).df()

        res_disp = res.copy()
        for c in res_disp.columns:
            cl = c.lower()
            if any(k in cl for k in ["aum", "inflow", "outflow", "net_flow", "revenue", "cost", "budget", "fee", "profit", "amount"]):
                if pd.api.types.is_numeric_dtype(res_disp[c]):
                    res_disp[c] = res_disp[c].map(fmt_eur)
            if "perf" in cl and pd.api.types.is_numeric_dtype(res_disp[c]):
                res_disp[c] = res_disp[c].map(fmt_pct)
            if "date" == cl:
                res_disp[c] = pd.to_datetime(res_disp[c]).dt.strftime("%d %b %Y")

        st.dataframe(res_disp, use_container_width=True)

        if res.shape[1] >= 2:
            date_cols = [c for c in res.columns if c.lower() == "date" or "date" in c.lower()]
            num_cols = [c for c in res.columns if pd.api.types.is_numeric_dtype(res[c])]
            cat_cols = [c for c in res.columns if c not in num_cols]

            if date_cols and num_cols:
                dcol = date_cols[0]
                ycol = num_cols[0]
                fig = px.line(res, x=dcol, y=ycol, template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360)
                st.plotly_chart(fig, use_container_width=True)
            elif len(cat_cols) >= 1 and len(num_cols) >= 1 and res.shape[1] == 2:
                xcol = cat_cols[0]
                ycol = num_cols[0]
                fig = px.bar(res, x=xcol, y=ycol, template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=360)
                st.plotly_chart(fig, use_container_width=True)
            elif any("status" in c.lower() for c in res.columns) and len(num_cols) == 1:
                scol = [c for c in res.columns if "status" in c.lower()][0]
                ycol = num_cols[0]
                fig = px.pie(res, names=scol, values=ycol, hole=0.6, template="plotly_dark")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=360)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Query failed: {e}")

