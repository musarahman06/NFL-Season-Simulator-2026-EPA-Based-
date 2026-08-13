import streamlit as st
import pandas as pd
import altair as alt
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ui_helpers import get_team_logo

st.set_page_config(page_title="Team Pages", page_icon="🏟️", layout="wide")


@st.cache_data
def load_data():
    # this file lives in app/pages/, so it takes TWO steps up to reach the project root, then into data/
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nfl.db")
    conn = sqlite3.connect(DB_PATH)
    final_season_games = pd.read_sql("SELECT * FROM final_season_games", conn)
    final_season_players = pd.read_sql("SELECT * FROM final_season_players", conn)
    teams_full = pd.read_sql("SELECT * FROM teams", conn)
    team_games_ref = pd.read_sql("SELECT team_id, season, team_abbr_current FROM team_games", conn)
    conn.close()
    return final_season_games, final_season_players, teams_full, team_games_ref


final_season_games, final_season_players, teams_full, team_games_ref = load_data()

abbr_to_id = dict(zip(teams_full["team_abbr"], teams_full["team_id"]))
team_abbr_lookup = team_games_ref.sort_values("season").groupby("team_id")["team_abbr_current"].last().to_dict()
# same relocated-franchise fix used throughout the project
abbr_overrides = {2510: "LAR", abbr_to_id["LV"]: "LV", abbr_to_id["LAC"]: "LAC"}
team_abbr_lookup.update(abbr_overrides)


def darken_hex(hex_color, factor=0.55):
    """Scales down each RGB channel to produce a darker shade of the same color."""
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return "#4d4d4d"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_team_colors(teams_full, team_id):
    """Primary color from the teams table + an auto-darkened secondary. Falls back to a
    neutral blue/dark-blue pair if this project's teams table doesn't have a color column
    under one of the common names -- check the actual column name in your `teams` table
    if this fallback is showing up and swap it into COLOR_COLUMN_CANDIDATES below."""
    COLOR_COLUMN_CANDIDATES = ["team_color", "primary_color", "color", "team_color1"]
    row = teams_full[teams_full["team_id"] == team_id]

    primary = None
    if not row.empty:
        for col in COLOR_COLUMN_CANDIDATES:
            if col in teams_full.columns:
                val = row.iloc[0].get(col)
                if pd.notna(val) and str(val).strip():
                    primary = str(val).strip()
                    break

    if primary is None:
        primary = "#1f77b4"  # fallback default
    elif not primary.startswith("#"):
        primary = f"#{primary}"

    secondary = darken_hex(primary)
    return primary, secondary


def render_box_score(game_id, game_row, players_df, team_abbr_lookup):
    """Renders both teams' passing/rushing/receiving lines for one game, inside whatever container calls it."""
    home_abbr = team_abbr_lookup.get(game_row["home_id"], "UNK")
    away_abbr = team_abbr_lookup.get(game_row["away_id"], "UNK")

    st.markdown(f"**Final: {away_abbr} {game_row['away_score']:.0f} - {game_row['home_score']:.0f} {home_abbr}**")

    col_home, col_away = st.columns(2)
    for col, tid, abbr in [(col_home, game_row["home_id"], home_abbr), (col_away, game_row["away_id"], away_abbr)]:
        with col:
            logo = get_team_logo(teams_full, tid)
            if logo:
                st.image(logo, width=40)
            st.markdown(f"**{abbr}**")
            team_stats = players_df[(players_df["game_id"] == game_id) & (players_df["team_id"] == tid)]
            team_stats = team_stats[team_stats["simulated_stat"] >= 1]

            pass_rows = team_stats[team_stats["stat_type"] == "passing"]
            if not pass_rows.empty:
                st.markdown("**Passing (QB)**")
                for _, r in pass_rows.iterrows():
                    st.text(f"{r['player_name']}: {r['completions']:.0f}/{r['attempts']:.0f}, {r['simulated_stat']:.0f} yds, {r['tds']:.0f} TD")

            rush_rows = team_stats[team_stats["stat_type"] == "rushing"].sort_values("simulated_stat", ascending=False)
            if not rush_rows.empty:
                st.markdown("**Rushing (RB)**")
                for _, r in rush_rows.iterrows():
                    st.text(f"{r['player_name']}: {r['carries']:.0f} car, {r['simulated_stat']:.0f} yds, {r['tds']:.0f} TD")

            rec_rows = team_stats[team_stats["stat_type"] == "receiving"].sort_values("simulated_stat", ascending=False)
            if not rec_rows.empty:
                st.markdown("**Receiving (WR/TE)**")
                for _, r in rec_rows.iterrows():
                    st.text(f"{r['player_name']}: {r['receptions']:.0f} rec, {r['simulated_stat']:.0f} yds, {r['tds']:.0f} TD")

            if pass_rows.empty and rush_rows.empty and rec_rows.empty:
                st.text("No notable stat lines this game.")


st.title("🏟️ Team Pages")
st.info("📊 **Data note:** this page reflects one detailed single-season simulation — not the 500-run Monte Carlo average used for playoff odds on the Home page.")

all_abbrs = sorted(set(team_abbr_lookup.values()))
selected_abbr = st.selectbox("Select a team", all_abbrs)
selected_id = next(tid for tid, abbr in team_abbr_lookup.items() if abbr == selected_abbr)

logo_col, name_col = st.columns([1, 8])
with logo_col:
    logo = get_team_logo(teams_full, selected_id)
    if logo:
        st.image(logo, width=70)
with name_col:
    st.markdown(f"## {selected_abbr}")

primary_color, secondary_color = get_team_colors(teams_full, selected_id)

team_games_this = final_season_games[
    (final_season_games["home_id"] == selected_id) | (final_season_games["away_id"] == selected_id)
].sort_values("week")

# --- record + points trend ---
wins, losses, ties = 0, 0, 0
trend_rows = []
for _, g in team_games_this.iterrows():
    is_home = g["home_id"] == selected_id
    team_score = g["home_score"] if is_home else g["away_score"]
    opp_score = g["away_score"] if is_home else g["home_score"]
    if team_score > opp_score:
        wins += 1
    elif team_score < opp_score:
        losses += 1
    else:
        ties += 1
    trend_rows.append({"Week": g["week"], "Points For": round(team_score), "Points Against": round(opp_score)})

trend_df = pd.DataFrame(trend_rows)

col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Simulated Record", f"{wins}-{losses}-{ties}")
    st.metric("Avg Points For", f"{trend_df['Points For'].mean():.1f}")
    st.metric("Avg Points Against", f"{trend_df['Points Against'].mean():.1f}")
with col2:
    st.markdown("**Points For vs. Points Against, by Week**")
    trend_long = trend_df.melt(id_vars="Week", value_vars=["Points For", "Points Against"], var_name="Metric", value_name="Points")
    trend_chart = alt.Chart(trend_long).mark_line(point=True).encode(
        x=alt.X("Week:O"),
        y=alt.Y("Points:Q"),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=["Points For", "Points Against"], range=[primary_color, secondary_color]), title=None),
    ).properties(height=280)
    st.altair_chart(trend_chart, use_container_width=True)

# --- schedule, with a box score expander per game ---
st.subheader("Schedule & Results")
for _, g in team_games_this.iterrows():
    is_home = g["home_id"] == selected_id
    team_score = g["home_score"] if is_home else g["away_score"]
    opp_score = g["away_score"] if is_home else g["home_score"]
    opp_id = g["away_id"] if is_home else g["home_id"]
    opp_abbr = team_abbr_lookup.get(opp_id, "UNK")
    matchup = f"vs {opp_abbr}" if is_home else f"@ {opp_abbr}"

    if team_score > opp_score:
        result = "W"
    elif team_score < opp_score:
        result = "L"
    else:
        result = "T"

    label = f"Week {g['week']:>2}  |  {matchup:<8}  |  {selected_abbr} {team_score:.0f}-{opp_score:.0f}  [{result}]"
    with st.expander(label):
        render_box_score(g["game_id"], g, final_season_players, team_abbr_lookup)

# --- combined stat leaders table (properly rounded to whole numbers) ---
st.subheader("Season Stat Leaders")
team_players = final_season_players[final_season_players["team_id"] == selected_id]

leaders = team_players.groupby(["player_name", "stat_type"]).agg(
    Yards=("simulated_stat", "sum"), TDs=("tds", "sum"), Games=("game_id", "nunique")
).reset_index().sort_values("Yards", ascending=False).rename(
    columns={"player_name": "Player", "stat_type": "Stat Type"}
)
leaders["Stat Type"] = leaders["Stat Type"].str.capitalize()
leaders["Yards"] = leaders["Yards"].round(0).astype(int)
leaders["TDs"] = leaders["TDs"].round(0).astype(int)

st.dataframe(leaders, hide_index=True, use_container_width=True)

# --- click into a player: per-week volume-stat bar chart + detail table ---
st.subheader("Player Game Log")
player_list = sorted(team_players["player_name"].unique())

if not player_list:
    st.info("No player stats available for this team.")
else:
    selected_player = st.selectbox("Select a player", player_list)
    player_data = team_players[team_players["player_name"] == selected_player].sort_values("week")
    stat_type = player_data["stat_type"].iloc[0]

    td_df = player_data[["week", "tds"]].copy()
    td_df["tds"] = td_df["tds"].round(0).astype(int)
    td_df = td_df.rename(columns={"week": "Week", "tds": "TDs"})

    st.markdown(f"**{selected_player} — Touchdowns by Week**")
    td_chart = alt.Chart(td_df).mark_line(point=True, color=primary_color).encode(
        x=alt.X("Week:O"),
        y=alt.Y("TDs:Q"),
    ).properties(height=280)
    st.altair_chart(td_chart, use_container_width=True)

    if stat_type == "passing":
        detail = player_data[["week", "completions", "attempts", "simulated_stat", "tds"]].copy()
        detail["completions"] = detail["completions"].round(0).astype(int)
        detail["attempts"] = detail["attempts"].round(0).astype(int)
        detail = detail.rename(columns={"week": "Week", "completions": "Comp", "attempts": "Att", "simulated_stat": "Yards", "tds": "TDs"})
    elif stat_type == "rushing":
        detail = player_data[["week", "carries", "simulated_stat", "tds"]].copy()
        detail["carries"] = detail["carries"].round(0).astype(int)
        detail = detail.rename(columns={"week": "Week", "carries": "Carries", "simulated_stat": "Yards", "tds": "TDs"})
    else:  # receiving
        detail = player_data[["week", "receptions", "simulated_stat", "tds"]].copy()
        detail["receptions"] = detail["receptions"].round(0).astype(int)
        detail = detail.rename(columns={"week": "Week", "receptions": "Rec", "simulated_stat": "Yards", "tds": "TDs"})

    detail["Yards"] = detail["Yards"].round(0).astype(int)
    detail["TDs"] = detail["TDs"].round(0).astype(int)

    st.dataframe(detail, hide_index=True, use_container_width=True)
