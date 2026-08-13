import streamlit as st
import pandas as pd
import os
import sys
import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import simulation_engine as sim
from ui_helpers import (
    render_playoff_bracket, render_division_standings, clean_player_stats_csv,
    generate_standings_pdf, get_team_logo, REPORTLAB_AVAILABLE
)

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False

st.set_page_config(page_title="Create Your Own Sim", page_icon="🎲", layout="wide")

st.title("🎲 Create Your Own Sim")
st.info("📊 **Data note:** this runs a brand-new single-season simulation live, using the same model and methods as the rest of the app.")
st.markdown(
    "Adjust each team's starting EPA below and/or reorder the power rankings, then run a full "
    "simulated season and playoffs. See the [How It Works](How_It_Works) page for what EPA means."
)

with st.spinner("Loading model and reference data (only happens once per session)..."):
    engine = sim.load_engine()


@st.cache_data
def get_base_team_states(_engine):
    return sim.init_team_states(_engine)


base_team_states = get_base_team_states(engine)

# ============================================================
# Power rankings -- drag-and-drop reorder
# ============================================================
st.subheader("Power Rankings")
st.caption(
    "Default order is based on NFL.com's preseason power rankings (published August 2026). "
    "Drag teams up or down to change how much the simulation favors them -- rank 1 gets the "
    "biggest boost, rank 32 the biggest penalty, and it's always a small nudge, never a guarantee."
)

default_order = sorted(sim.POWER_RANKINGS, key=lambda a: sim.POWER_RANKINGS[a])

if SORTABLES_AVAILABLE:
    with st.expander("Show team logos (reference while dragging)", expanded=False):
        legend_cols = st.columns(8)
        for i, abbr in enumerate(default_order):
            tid = engine.abbr_to_id[abbr]
            logo = get_team_logo(engine.teams_full, tid)
            with legend_cols[i % 8]:
                if logo:
                    st.image(logo, width=28)
                st.caption(abbr)

    sorted_order = sort_items(default_order, direction="vertical", key="power_ranking_sort")
    custom_power_rankings = {abbr: i + 1 for i, abbr in enumerate(sorted_order)}
else:
    st.warning("Drag-and-drop reordering needs the `streamlit-sortables` package. "
               "Run `pip install streamlit-sortables` to enable it. Using the default NFL.com order for now.")
    custom_power_rankings = dict(sim.POWER_RANKINGS)

power_rank_by_id = {engine.abbr_to_id[abbr]: rank for abbr, rank in custom_power_rankings.items()}
custom_team_strength = sim.compute_team_strength(engine.abbr_to_id, custom_power_rankings)

# ============================================================
# EPA editor -- with team logos
# ============================================================
rows = []
for tid in engine.all_team_ids:
    rows.append({
        "team_id": tid,
        "Logo": get_team_logo(engine.teams_full, tid),
        "Team": engine.team_abbr_lookup.get(tid, "UNK"),
        "Off EPA (roll3)": round(sim.get_rolling_avg(base_team_states, tid, "total_epa", "roll3"), 2),
        "Off EPA (roll8)": round(sim.get_rolling_avg(base_team_states, tid, "total_epa", "roll8"), 2),
        "Def EPA (roll3)": round(sim.get_rolling_avg(base_team_states, tid, "def_total_epa", "roll3"), 2),
        "Def EPA (roll8)": round(sim.get_rolling_avg(base_team_states, tid, "def_total_epa", "roll8"), 2),
    })

epa_df = pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)
team_ids_ordered = epa_df["team_id"].tolist()

st.subheader("Starting EPA (editable)")
st.caption("Hover the ⓘ on each column header for what it means and what raising or lowering it does.")

epa_column_config = {
    "Logo": st.column_config.ImageColumn("Logo", help="Team logo"),
    "Off EPA (roll3)": st.column_config.NumberColumn(
        help="Offensive EPA (Expected Points Added) per game, averaged over this team's last "
             "3 real games. Higher = a more efficient offense that's more likely to score. "
             "Raising this number makes the model expect more points and wins from this team, "
             "especially in the first ~9 games of the season before real simulated results take over."
    ),
    "Off EPA (roll8)": st.column_config.NumberColumn(
        help="Same idea as roll3, but averaged over the last 8 real games -- a steadier read on "
             "the offense. This one carries more lasting weight across the season than roll3, "
             "since it doesn't fade out as quickly once real games start being simulated."
    ),
    "Def EPA (roll3)": st.column_config.NumberColumn(
        help="Defensive EPA ALLOWED per game, averaged over this team's last 3 real games. "
             "Lower (more negative) = a stingier defense that gives up fewer expected points. "
             "Raising this number makes the defense worse (allows more); lowering it makes the "
             "defense better (allows less)."
    ),
    "Def EPA (roll8)": st.column_config.NumberColumn(
        help="Same idea as roll3, but averaged over the last 8 real games -- a steadier, more "
             "lasting read on this defense across the season."
    ),
}

edited_df = st.data_editor(
    epa_df.drop(columns=["team_id"]),
    hide_index=True,
    use_container_width=True,
    disabled=["Team", "Logo"],
    num_rows="fixed",
    column_config=epa_column_config,
    key="epa_editor",
)

run_clicked = st.button("🎲 Run My Simulation", type="primary")

if run_clicked:
    overrides = {}
    for i, tid in enumerate(team_ids_ordered):
        row = edited_df.iloc[i]
        overrides[tid] = {
            "off_roll3": row["Off EPA (roll3)"], "off_roll8": row["Off EPA (roll8)"],
            "def_roll3": row["Def EPA (roll3)"], "def_roll8": row["Def EPA (roll8)"],
        }
    custom_team_states = sim.apply_epa_overrides(base_team_states, overrides)
    epa_lookup_for_pdf = {tid: overrides[tid]["off_roll8"] for tid in team_ids_ordered}

    with st.spinner("Simulating the full 2026 regular season (272 games)..."):
        final_states, games_df, players_df = sim.simulate_one_season_full(
            engine, custom_team_states, team_strength=custom_team_strength
        )

    home_r = games_df[["home_id", "away_id", "home_score", "away_score"]].rename(
        columns={"home_id": "team_id", "away_id": "opp_id", "home_score": "team_score", "away_score": "opp_score"})
    away_r = games_df[["home_id", "away_id", "home_score", "away_score"]].rename(
        columns={"away_id": "team_id", "home_id": "opp_id", "away_score": "team_score", "home_score": "opp_score"})
    team_games_long = pd.concat([home_r, away_r], ignore_index=True)
    team_games_long["win"] = (team_games_long["team_score"] > team_games_long["opp_score"]).astype(int)
    team_games_long["loss"] = (team_games_long["team_score"] < team_games_long["opp_score"]).astype(int)
    team_games_long["tie"] = (team_games_long["team_score"] == team_games_long["opp_score"]).astype(int)

    record = team_games_long.groupby("team_id").agg(
        wins=("win", "sum"), losses=("loss", "sum"), ties=("tie", "sum")
    ).reset_index()
    record = record.merge(engine.team_conf_div, on="team_id")
    avg_pts = team_games_long.groupby("team_id")["team_score"].mean().reset_index().rename(columns={"team_score": "avg_points"})
    record = record.merge(avg_pts, on="team_id")
    record["team_abbr"] = record["team_id"].map(engine.team_abbr_lookup)
    record["Record"] = record["wins"].astype(str) + "-" + record["losses"].astype(str) + "-" + record["ties"].astype(str)

    seeds = sim.get_playoff_seeds(record)
    seeds["conf"] = seeds["team_conf"]

    with st.spinner("Simulating the playoffs..."):
        playoff_games_df, playoff_players_df, sb_winner, afc_champ, nfc_champ = sim.simulate_full_playoffs(
            engine, final_states, seeds, team_strength=custom_team_strength
        )

    if "sim_history" not in st.session_state:
        st.session_state["sim_history"] = []

    st.session_state["sim_history"].append({
        "Sim #": len(st.session_state["sim_history"]) + 1,
        "Super Bowl Winner": engine.team_abbr_lookup.get(sb_winner, "UNK"),
        "AFC Champion": engine.team_abbr_lookup.get(afc_champ, "UNK"),
        "NFC Champion": engine.team_abbr_lookup.get(nfc_champ, "UNK"),
        "Run At": datetime.datetime.now().strftime("%H:%M:%S"),
    })

    st.session_state["custom_sim"] = {
        "games_df": games_df, "players_df": players_df, "record": record, "seeds": seeds,
        "playoff_games_df": playoff_games_df, "playoff_players_df": playoff_players_df,
        "power_rank_lookup": power_rank_by_id, "epa_lookup": epa_lookup_for_pdf,
    }
    st.success("Simulation complete! Scroll down for results.")

if st.session_state.get("sim_history"):
    st.divider()
    st.subheader("Past Sims (This Session)")
    st.caption("Every simulation you've run since opening this page. Refreshing the browser clears this list.")
    history_df = pd.DataFrame(st.session_state["sim_history"]).sort_values("Sim #", ascending=False)
    st.dataframe(history_df, hide_index=True, use_container_width=True)

if "custom_sim" in st.session_state:
    results = st.session_state["custom_sim"]

    st.divider()
    st.header("Your Simulated Season")

    seed_lookup = dict(zip(results["seeds"]["team_id"], results["seeds"]["seed"]))

    st.subheader("Full Season Standings by Division")
    render_division_standings(results["record"], engine.team_abbr_lookup, engine.teams_full, seed_lookup=seed_lookup)

    st.subheader("Playoff Bracket")
    render_playoff_bracket(
        results["playoff_games_df"], results["playoff_players_df"],
        engine.team_abbr_lookup, engine.teams_full, seed_lookup=seed_lookup
    )

    st.subheader("Downloads")
    col1, col2, col3 = st.columns(3)
    with col1:
        clean_regular = clean_player_stats_csv(results["players_df"], engine.team_abbr_lookup, time_col="week", time_label="Week")
        csv_regular = clean_regular.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Regular season player stats (CSV)", csv_regular, "custom_sim_regular_season_players.csv", "text/csv")
    with col2:
        clean_playoff = clean_player_stats_csv(results["playoff_players_df"], engine.team_abbr_lookup, time_col="round", time_label="Round")
        csv_playoff = clean_playoff.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Playoff player stats (CSV)", csv_playoff, "custom_sim_playoff_players.csv", "text/csv")
    with col3:
        if REPORTLAB_AVAILABLE:
            with st.spinner("Building PDF (fetching team logos)..."):
                pdf_bytes = generate_standings_pdf(
                    results["record"], engine.team_abbr_lookup, engine.teams_full,
                    seed_lookup=seed_lookup, power_rank_lookup=results["power_rank_lookup"],
                    epa_lookup=results["epa_lookup"], title="Your Simulated Season Standings"
                )
            st.download_button("⬇️ Standings breakdown (PDF)", pdf_bytes, "custom_sim_standings.pdf", "application/pdf")
        else:
            st.button("⬇️ Standings breakdown (PDF)", disabled=True, help="Run `pip install reportlab` to enable PDF export.")
