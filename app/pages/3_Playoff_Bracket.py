import streamlit as st
import pandas as pd
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ui_helpers import render_playoff_bracket, render_division_standings

st.set_page_config(page_title="Playoff Bracket", page_icon="🏆", layout="wide")


@st.cache_data
def load_data():
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nfl.db")
    conn = sqlite3.connect(DB_PATH)
    final_season_games = pd.read_sql("SELECT * FROM final_season_games", conn)
    final_playoff_games = pd.read_sql("SELECT * FROM final_playoff_games", conn)
    final_playoff_players = pd.read_sql("SELECT * FROM final_playoff_players", conn)
    final_playoff_seeds = pd.read_sql("SELECT * FROM final_playoff_seeds", conn)
    teams_full = pd.read_sql("SELECT * FROM teams", conn)
    team_games_ref = pd.read_sql("SELECT team_id, season, team_abbr_current FROM team_games", conn)
    conn.close()
    return (final_season_games, final_playoff_games, final_playoff_players,
            final_playoff_seeds, teams_full, team_games_ref)


(final_season_games, final_playoff_games, final_playoff_players,
 final_playoff_seeds, teams_full, team_games_ref) = load_data()

abbr_to_id = dict(zip(teams_full["team_abbr"], teams_full["team_id"]))
team_abbr_lookup = team_games_ref.sort_values("season").groupby("team_id")["team_abbr_current"].last().to_dict()
abbr_overrides = {2510: "LAR", abbr_to_id["LV"]: "LV", abbr_to_id["LAC"]: "LAC"}
team_abbr_lookup.update(abbr_overrides)

team_conf_div = teams_full[["team_id", "team_conf", "team_division"]].drop_duplicates(subset="team_id")

# seed lookup: {team_id: seed} for the 14 teams that made the playoffs this season
seed_lookup = dict(zip(final_playoff_seeds["team_id"], final_playoff_seeds["seed"]))

st.title("🏆 Playoff Bracket")
st.info("📊 **Data note:** this reflects one detailed single-season simulation — not the 500-run Monte Carlo average used for playoff odds on the Home page.")
st.caption("Click any matchup to see the full box score for both teams.")

render_playoff_bracket(final_playoff_games, final_playoff_players, team_abbr_lookup, teams_full, seed_lookup=seed_lookup)

st.divider()
st.header("Full Season Standings by Division")
st.caption("Every team's simulated regular-season record. Seed numbers shown for the 14 teams that made the playoffs.")

# build full-season records from every regular-season game
home_r = final_season_games[["home_id", "away_id", "home_score", "away_score"]].rename(
    columns={"home_id": "team_id", "away_id": "opp_id", "home_score": "team_score", "away_score": "opp_score"})
away_r = final_season_games[["home_id", "away_id", "home_score", "away_score"]].rename(
    columns={"away_id": "team_id", "home_id": "opp_id", "away_score": "team_score", "home_score": "opp_score"})
team_games_long = pd.concat([home_r, away_r], ignore_index=True)
team_games_long["win"] = (team_games_long["team_score"] > team_games_long["opp_score"]).astype(int)
team_games_long["loss"] = (team_games_long["team_score"] < team_games_long["opp_score"]).astype(int)
team_games_long["tie"] = (team_games_long["team_score"] == team_games_long["opp_score"]).astype(int)

record = team_games_long.groupby("team_id").agg(
    wins=("win", "sum"), losses=("loss", "sum"), ties=("tie", "sum")
).reset_index()
record = record.merge(team_conf_div, on="team_id")
avg_pts = team_games_long.groupby("team_id")["team_score"].mean().reset_index().rename(columns={"team_score": "avg_points"})
record = record.merge(avg_pts, on="team_id")

render_division_standings(record, team_abbr_lookup, teams_full, seed_lookup=seed_lookup)
