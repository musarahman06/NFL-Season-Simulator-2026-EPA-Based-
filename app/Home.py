import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="2026 NFL Season Simulator", page_icon="🏈", layout="wide")


@st.cache_data
def load_data():
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nfl.db")
    conn = sqlite3.connect(DB_PATH)
    season_summary = pd.read_sql("SELECT * FROM season_simulation_summary", conn)
    season_records = pd.read_sql("SELECT * FROM season_records_by_sim", conn)
    teams_full = pd.read_sql("SELECT * FROM teams", conn)
    conn.close()
    return season_summary, season_records, teams_full


season_summary, season_records, teams_full = load_data()
team_conf_div = teams_full[["team_id", "team_conf", "team_division"]].drop_duplicates(subset="team_id")

# build a display-ready standings table: merge conf/division + average record onto the Monte Carlo summary
avg_records = season_records.groupby("team_id").agg(
    avg_wins=("wins", "mean"), avg_losses=("losses", "mean")
).reset_index()
standings = season_summary.merge(team_conf_div, on="team_id").merge(avg_records, on="team_id")
standings["record"] = standings["avg_wins"].round(1).astype(str) + "-" + standings["avg_losses"].round(1).astype(str)

st.title("🏈 2026 NFL Season Simulator")
st.caption("Built on 500 Monte Carlo season simulations + one detailed single-season run")

tab1, tab2 = st.tabs(["Division Standings", "League-Wide Odds"])

with tab1:
    for conf in ["AFC", "NFC"]:
        st.subheader(conf)
        conf_data = standings[standings["team_conf"] == conf]
        divisions = sorted(conf_data["team_division"].unique())
        cols = st.columns(len(divisions))
        for col, div in zip(cols, divisions):
            with col:
                st.markdown(f"**{div}**")
                div_data = conf_data[conf_data["team_division"] == div].sort_values("playoff_pct", ascending=False)
                display = div_data[["team_abbr", "record", "playoff_pct", "sb_win_pct"]].rename(
                    columns={"team_abbr": "Team", "record": "Record", "playoff_pct": "Playoff %", "sb_win_pct": "SB Win %"}
                )
                st.dataframe(display, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("Every team's playoff and Super Bowl odds across all 500 simulations, ranked by Super Bowl win probability.")
    display = standings.sort_values("sb_win_pct", ascending=False)[
        ["team_abbr", "record", "playoff_pct", "sb_win_pct"]
    ].rename(columns={"team_abbr": "Team", "record": "Avg Record", "playoff_pct": "Playoff %", "sb_win_pct": "Super Bowl Win %"})
    st.dataframe(display, hide_index=True, use_container_width=True)

st.sidebar.info("Use the pages above to explore team pages, box scores, the playoff bracket, stat leaderboards, matchup predictions, and run your own simulation.")
