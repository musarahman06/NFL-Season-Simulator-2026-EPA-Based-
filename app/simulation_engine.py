"""
Shared simulation engine for the Streamlit app.

This is a direct port of the core logic from notebook 05 (feature building,
prediction, game/season/playoff simulation, player stat distribution) into a
standalone module, so the app doesn't depend on any notebook being open or
re-run. All known bugs from 05's cleanup are preserved as fixed here too.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import copy
import os
from collections import deque
from types import SimpleNamespace

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "..", "data", "nfl.db")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_point_diff_v1.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_cols_v1.pkl")

STAT_NAMES = ["points_for", "points_against", "total_epa", "def_total_epa", "total_yards", "def_total_yards"]
MODEL_MAE = 10.3
LEAGUE_AVG_TOTAL = 45.67
TD_PROBABILITY = 0.58
MAX_WEEK1_NUDGE = 3.0
MAX_PERMANENT_PULL = 1.0

POWER_RANKINGS = {
    "LA": 1, "DEN": 2, "SEA": 3, "PHI": 4, "BUF": 5, "BAL": 6, "JAX": 7, "CIN": 8,
    "HOU": 9, "CHI": 10, "DAL": 11, "NE": 12, "GB": 13, "DET": 14, "KC": 15, "SF": 16,
    "LAC": 17, "MIN": 18, "IND": 19, "PIT": 20, "NO": 21, "TB": 22, "CAR": 23, "TEN": 24,
    "ATL": 25, "ARI": 26, "NYG": 27, "CLE": 28, "WAS": 29, "NYJ": 30, "LV": 31, "MIA": 32,
}


@st.cache_resource
def load_engine():
    """Loads and precomputes every static table the simulation needs. Cached so this
    only runs once per app session, not on every rerun."""
    conn = sqlite3.connect(DB_PATH)
    xgb_model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    team_games = pd.read_sql("SELECT * FROM team_games", conn)
    games = pd.read_sql("SELECT * FROM games", conn)
    teams_full = pd.read_sql("SELECT * FROM teams", conn)
    pgs = pd.read_sql("SELECT * FROM player_game_stats", conn)
    depth_current = pd.read_sql("SELECT * FROM depth_charts_current", conn)
    conn.close()

    abbr_to_id = dict(zip(teams_full["team_abbr"], teams_full["team_id"]))
    team_games_ref = team_games[["team_id", "season", "team_abbr_current"]]
    team_abbr_lookup = team_games_ref.sort_values("season").groupby("team_id")["team_abbr_current"].last().to_dict()
    team_abbr_lookup.update({2510: "LAR", abbr_to_id["LV"]: "LV", abbr_to_id["LAC"]: "LAC"})

    team_conf_div = teams_full[["team_id", "team_conf", "team_division"]].drop_duplicates(subset="team_id")

    team_strength = {abbr_to_id[abbr]: (16.5 - rank) / 15.5 for abbr, rank in POWER_RANKINGS.items()}

    # --- rebuild EPA (NaN-safe) ---
    stats_mask = team_games["passing_yards"].notna()
    team_games["total_epa"] = np.nan
    team_games["def_total_epa"] = np.nan
    team_games.loc[stats_mask, "total_epa"] = (
        team_games.loc[stats_mask, "passing_epa"].fillna(0)
        + team_games.loc[stats_mask, "rushing_epa"].fillna(0)
        + team_games.loc[stats_mask, "receiving_epa"].fillna(0)
    )
    team_games.loc[stats_mask, "def_total_epa"] = (
        team_games.loc[stats_mask, "def_passing_epa"].fillna(0)
        + team_games.loc[stats_mask, "def_rushing_epa"].fillna(0)
        + team_games.loc[stats_mask, "def_receiving_epa"].fillna(0)
    )

    # --- rolling features (with correct 2025/2026 carry-forward) ---
    team_games = team_games.sort_values(["team_id", "season", "week"]).reset_index(drop=True)
    grouped = team_games.groupby("team_id")
    for col in ["points_for", "points_against"]:
        shifted = grouped[col].shift(1)
        team_games[f"{col}_roll3"] = shifted.groupby(team_games["team_id"]).transform(lambda x: x.rolling(3, min_periods=1).mean())
        team_games[f"{col}_roll8"] = shifted.groupby(team_games["team_id"]).transform(lambda x: x.rolling(8, min_periods=1).mean())
        team_games[f"{col}_season"] = shifted.groupby(team_games["season"].astype(str) + "_" + team_games["team_id"].astype(str)).transform(lambda x: x.expanding(min_periods=1).mean())

    for col in ["total_epa", "def_total_epa", "total_yards", "def_total_yards"]:
        real_only = team_games[col].where(stats_mask)
        shifted = real_only.groupby(team_games["team_id"]).shift(1)
        team_games[f"{col}_roll3"] = shifted.groupby(team_games["team_id"]).transform(lambda x: x.rolling(3, min_periods=1).mean())
        team_games[f"{col}_roll8"] = shifted.groupby(team_games["team_id"]).transform(lambda x: x.rolling(8, min_periods=1).mean())
        team_games[f"{col}_season"] = shifted.groupby(team_games["season"].astype(str) + "_" + team_games["team_id"].astype(str)).transform(lambda x: x.expanding(min_periods=1).mean())
        for roll_col in [f"{col}_roll3", f"{col}_roll8", f"{col}_season"]:
            last_real_idx = team_games[stats_mask].groupby("team_id").tail(1).index
            last_real_values = team_games.loc[last_real_idx, ["team_id", roll_col]].set_index("team_id")[roll_col]
            future_mask = ~stats_mask
            team_games.loc[future_mask, roll_col] = team_games.loc[future_mask, "team_id"].map(last_real_values)

    # --- player stat distribution reference tables ---
    pgs_all = pgs.copy()
    pgs_all["team_id"] = pgs_all["recent_team"].map(abbr_to_id)

    def compute_rank_shares(pgs_df, stat_col, position_filter=None):
        df = pgs_df.copy()
        if position_filter:
            df = df[df["position"].isin(position_filter)]
        season_totals = df.groupby(["team_id", "season", "player_id"])[stat_col].sum().reset_index()
        season_totals["rank"] = season_totals.groupby(["team_id", "season"])[stat_col].rank(method="first", ascending=False)
        team_season_totals = season_totals.groupby(["team_id", "season"])[stat_col].transform("sum")
        season_totals["share"] = season_totals[stat_col] / team_season_totals
        return season_totals.groupby("rank")["share"].mean()

    rush_shares = compute_rank_shares(pgs_all, "rushing_yards", position_filter=["RB"])
    wr_shares = compute_rank_shares(pgs_all, "receiving_yards", position_filter=["WR"])
    te_shares = compute_rank_shares(pgs_all, "receiving_yards", position_filter=["TE"])
    pass_shares = compute_rank_shares(pgs_all, "passing_yards", position_filter=["QB"])

    wr_te_split = pgs_all[pgs_all["position"].isin(["WR", "TE"])].groupby(["team_id", "season", "position"])["receiving_yards"].sum().reset_index()
    position_totals = wr_te_split.groupby("position")["receiving_yards"].sum()
    position_share = position_totals / position_totals.sum()

    depth_current = depth_current.copy()
    depth_current["team_id"] = depth_current["team"].map(abbr_to_id)
    depth_current_skill = depth_current[depth_current["pos_abb"].isin(["QB", "RB", "WR", "TE"])].copy()
    current_qbs = depth_current_skill[depth_current_skill["pos_abb"] == "QB"]
    current_rbs = depth_current_skill[depth_current_skill["pos_abb"] == "RB"]
    current_wrs = depth_current_skill[depth_current_skill["pos_abb"] == "WR"]
    current_tes = depth_current_skill[depth_current_skill["pos_abb"] == "TE"]

    real_2020_2024 = team_games[(team_games["season"].between(2020, 2024)) & (team_games["passing_yards"].notna())]
    team_split = real_2020_2024.groupby("team_id").agg(
        total_pass_yards=("passing_yards", "sum"), total_rush_yards=("rushing_yards", "sum")
    )
    team_split["pass_share"] = team_split["total_pass_yards"] / (team_split["total_pass_yards"] + team_split["total_rush_yards"])

    real_recent_full = pgs[pgs["season"].between(2020, 2024)].copy()
    real_recent_full["team_id"] = real_recent_full["recent_team"].map(abbr_to_id)
    team_td_counts = real_recent_full.groupby("team_id").agg(
        total_pass_tds=("passing_tds", "sum"), total_rush_tds=("rushing_tds", "sum")
    )
    team_td_counts["rush_td_share_correct"] = team_td_counts["total_rush_tds"] / (team_td_counts["total_rush_tds"] + team_td_counts["total_pass_tds"])

    real_recent = pgs[pgs["season"].between(2020, 2024)].copy()
    real_recent["team_id"] = real_recent["recent_team"].map(abbr_to_id)
    rush_rates = real_recent.groupby("team_id").apply(
        lambda x: pd.Series({
            "yards_per_carry": x["rushing_yards"].sum() / x["carries"].sum() if x["carries"].sum() > 0 else 4.2,
            "rush_td_rate": x["rushing_tds"].sum() / x["rushing_yards"].sum() if x["rushing_yards"].sum() > 0 else 0.01
        })
    )
    pass_rates = real_recent.groupby("team_id").apply(
        lambda x: pd.Series({
            "comp_pct": x["completions"].sum() / x["attempts"].sum() if x["attempts"].sum() > 0 else 0.63,
            "pass_td_rate": x["passing_tds"].sum() / x["passing_yards"].sum() if x["passing_yards"].sum() > 0 else 0.02
        })
    )
    rec_rates = real_recent.groupby("team_id").apply(
        lambda x: pd.Series({
            "yards_per_target": x["receiving_yards"].sum() / x["targets"].sum() if x["targets"].sum() > 0 else 7.5,
            "catch_rate": x["receptions"].sum() / x["targets"].sum() if x["targets"].sum() > 0 else 0.63,
        })
    )
    team_td_rates = team_games[
        (team_games["season"].between(2020, 2024)) & (team_games["passing_tds"].notna())
    ].groupby("team_id").apply(lambda x: (x["passing_tds"] + x["rushing_tds"]).mean()).rename("avg_tds_per_game")

    # --- schedule prep ---
    schedule_2026 = games[games["season"] == 2026].copy()
    schedule_2026["home_id"] = schedule_2026["home_team"].map(abbr_to_id)
    schedule_2026["away_id"] = schedule_2026["away_team"].map(abbr_to_id)
    schedule_2026 = schedule_2026.sort_values(["week", "game_id"]).reset_index(drop=True)
    stadium_roof = team_games.dropna(subset=["roof"]).groupby("stadium")["roof"].agg(lambda x: x.mode()[0])

    def get_roof(row):
        if pd.notna(row["roof"]):
            return row["roof"]
        return stadium_roof.get(row["stadium"], "outdoors")

    schedule_2026["roof"] = schedule_2026.apply(get_roof, axis=1)

    real_games = team_games[team_games["points_for"].notna()]
    team_scoring_std = real_games.groupby("team_id")["points_for"].std().to_dict()
    team_scoring_std_default = real_games["points_for"].std()

    return SimpleNamespace(
        xgb_model=xgb_model, feature_cols=feature_cols, team_games=team_games,
        abbr_to_id=abbr_to_id, team_abbr_lookup=team_abbr_lookup, team_conf_div=team_conf_div,
        team_strength=team_strength, schedule_2026=schedule_2026, all_team_ids=team_games["team_id"].unique(),
        team_scoring_std=team_scoring_std, team_scoring_std_default=team_scoring_std_default,
        rush_shares=rush_shares, wr_shares=wr_shares, te_shares=te_shares, pass_shares=pass_shares,
        position_share=position_share, current_qbs=current_qbs, current_rbs=current_rbs,
        current_wrs=current_wrs, current_tes=current_tes, team_split=team_split,
        team_td_counts=team_td_counts, rush_rates=rush_rates, pass_rates=pass_rates,
        rec_rates=rec_rates, team_td_rates=team_td_rates, teams_full=teams_full,
    )


def init_team_states(engine):
    states = {}
    for tid in engine.all_team_ids:
        team_hist = engine.team_games[engine.team_games["team_id"] == tid].sort_values(["season", "week"])
        states[tid] = {}
        for stat in STAT_NAMES:
            real_vals = team_hist[team_hist[stat].notna()][stat].tolist()
            last3 = real_vals[-3:] if len(real_vals) >= 1 else [0]
            last8 = real_vals[-8:] if len(real_vals) >= 1 else [0]
            states[tid][stat] = {
                "roll3_deque": deque(last3, maxlen=3),
                "roll8_deque": deque(last8, maxlen=8),
                "season_vals": []
            }
    return states


def apply_epa_overrides(team_states, overrides):
    """overrides: {team_id: {"off_roll3": v, "off_roll8": v, "def_roll3": v, "def_roll8": v}}
    Reseeds the EPA deques so their mean equals the user-provided value, leaving every
    other stat (points, yards) at its real historical seed."""
    ts = copy.deepcopy(team_states)
    for tid, vals in overrides.items():
        if tid not in ts:
            continue
        if "off_roll3" in vals:
            ts[tid]["total_epa"]["roll3_deque"] = deque([vals["off_roll3"]] * 3, maxlen=3)
        if "off_roll8" in vals:
            ts[tid]["total_epa"]["roll8_deque"] = deque([vals["off_roll8"]] * 8, maxlen=8)
        if "def_roll3" in vals:
            ts[tid]["def_total_epa"]["roll3_deque"] = deque([vals["def_roll3"]] * 3, maxlen=3)
        if "def_roll8" in vals:
            ts[tid]["def_total_epa"]["roll8_deque"] = deque([vals["def_roll8"]] * 8, maxlen=8)
    return ts


def get_rolling_avg(team_states, tid, stat, window):
    if window == "roll3":
        vals = list(team_states[tid][stat]["roll3_deque"])
    elif window == "roll8":
        vals = list(team_states[tid][stat]["roll8_deque"])
    elif window == "season":
        vals = team_states[tid][stat]["season_vals"]
        if len(vals) == 0:
            vals = list(team_states[tid][stat]["roll8_deque"])
    return np.mean(vals) if len(vals) > 0 else 0


def compute_team_strength(abbr_to_id, power_rankings):
    """Same formula as load_engine's default, but usable with any custom rank ordering
    (e.g. a user's drag-and-drop reordered power rankings)."""
    return {abbr_to_id[abbr]: (16.5 - rank) / 15.5 for abbr, rank in power_rankings.items()}


def get_power_nudge(engine, team_id, opp_id, team_states, team_strength=None):
    ts = team_strength if team_strength is not None else engine.team_strength
    strength_gap = ts.get(team_id, 0) - ts.get(opp_id, 0)
    games_played = len(team_states[team_id]["points_for"]["season_vals"])
    decay = max(0, 1 - games_played / 9)
    return MAX_WEEK1_NUDGE * (strength_gap / 2) * decay + MAX_PERMANENT_PULL * (strength_gap / 2)


def build_features(team_states, team_id, opp_id, is_home, div_game, rest_days, opp_rest_days,
                    temp, wind, is_indoor, qb_out, rb_out, wr_out):
    row = {
        "div_game": div_game, "temp": temp, "wind": wind, "rest_days": rest_days,
        "opp_rest_days": opp_rest_days, "is_home": is_home, "rest_advantage": rest_days - opp_rest_days,
        "short_week": int(rest_days < 6), "is_indoor": is_indoor, "qb_out": qb_out,
        "rb_starter_out": rb_out, "wr_starter_out": wr_out,
    }
    for stat in ["points_for", "points_against", "total_epa", "def_total_epa", "total_yards", "def_total_yards"]:
        row[f"{stat}_roll3"] = get_rolling_avg(team_states, team_id, stat, "roll3")
        row[f"{stat}_roll8"] = get_rolling_avg(team_states, team_id, stat, "roll8")
        row[f"{stat}_season"] = get_rolling_avg(team_states, team_id, stat, "season")
    row["opp_def_epa_roll3"] = get_rolling_avg(team_states, opp_id, "def_total_epa", "roll3")
    row["opp_def_epa_roll8"] = get_rolling_avg(team_states, opp_id, "def_total_epa", "roll8")
    row["opp_def_epa_season"] = get_rolling_avg(team_states, opp_id, "def_total_epa", "season")
    row["opp_off_epa_roll3"] = get_rolling_avg(team_states, opp_id, "total_epa", "roll3")
    row["opp_off_epa_roll8"] = get_rolling_avg(team_states, opp_id, "total_epa", "roll8")
    row["opp_off_epa_season"] = get_rolling_avg(team_states, opp_id, "total_epa", "season")
    row["adj_epa_roll3"] = row["total_epa_roll3"] - row["opp_def_epa_roll3"]
    row["adj_epa_roll8"] = row["total_epa_roll8"] - row["opp_def_epa_roll8"]
    row["adj_epa_season"] = row["total_epa_season"] - row["opp_def_epa_season"]
    row["adj_def_epa_roll3"] = row["opp_off_epa_roll3"] - row["def_total_epa_roll3"]
    row["adj_def_epa_roll8"] = row["opp_off_epa_roll8"] - row["def_total_epa_roll8"]
    row["adj_def_epa_season"] = row["opp_off_epa_season"] - row["def_total_epa_season"]
    return row


def predict_game(engine, team_states, team_id, opp_id, is_home, div_game, rest_days, opp_rest_days,
                  temp, wind, is_indoor, qb_out, rb_out, wr_out, mae=MODEL_MAE, team_strength=None):
    row = build_features(team_states, team_id, opp_id, is_home, div_game, rest_days, opp_rest_days,
                          temp, wind, is_indoor, qb_out, rb_out, wr_out)
    X = pd.DataFrame([row])[engine.feature_cols]
    predicted_diff = engine.xgb_model.predict(X)[0]
    predicted_diff += get_power_nudge(engine, team_id, opp_id, team_states, team_strength=team_strength)
    simulated_diff = predicted_diff + np.random.normal(loc=0, scale=mae)
    return simulated_diff, predicted_diff


def get_team_scoring_std(engine, tid):
    val = engine.team_scoring_std.get(tid, np.nan)
    return val if not np.isnan(val) else engine.team_scoring_std_default


def simulate_score_from_avg(avg_points):
    points_per_drive = TD_PROBABILITY * 7 + (1 - TD_PROBABILITY) * 3
    est_drives = max(round(avg_points / points_per_drive), 0)
    est_drives = max(round(np.random.normal(est_drives, 1.2)), 0)  # round(), not int() -- avoids low-bias drift
    total = 0
    for _ in range(est_drives):
        total += 7 if np.random.random() < TD_PROBABILITY else 3
    return total


def simulate_game_result(engine, team_states, team_id, opp_id, simulated_diff):
    team_off_avg = get_rolling_avg(team_states, team_id, "points_for", "roll8")
    team_def_avg = get_rolling_avg(team_states, team_id, "points_against", "roll8")
    opp_off_avg = get_rolling_avg(team_states, opp_id, "points_for", "roll8")
    opp_def_avg = get_rolling_avg(team_states, opp_id, "points_against", "roll8")

    home_expected = (team_off_avg + opp_def_avg) / 2
    away_expected = (opp_off_avg + team_def_avg) / 2
    expected_total = home_expected + away_expected
    blended_total = (expected_total + LEAGUE_AVG_TOTAL) / 2

    team_std = get_team_scoring_std(engine, team_id)
    opp_std = get_team_scoring_std(engine, opp_id)
    combined_std = np.sqrt(team_std**2 + opp_std**2)
    max(blended_total + np.random.normal(0, combined_std), 6)  # total_points computed for parity w/ 05, unused directly below

    team_points = simulate_score_from_avg(home_expected)
    opp_points = simulate_score_from_avg(away_expected)
    actual_diff = team_points - opp_points
    diff_error = simulated_diff - actual_diff
    adjustment = round(diff_error / 2)
    team_points = max(team_points + adjustment, 0)
    opp_points = max(opp_points - adjustment, 0)

    team_epa = simulated_diff * 0.5 + np.random.normal(0, 8)
    opp_epa = -simulated_diff * 0.5 + np.random.normal(0, 8)
    team_yards = 350 + simulated_diff * 3 + np.random.normal(0, 30)
    opp_yards = 350 - simulated_diff * 3 + np.random.normal(0, 30)

    return {
        "points_for": team_points, "points_against": opp_points,
        "total_epa": team_epa, "def_total_epa": opp_epa,
        "total_yards": max(team_yards, 0), "def_total_yards": max(opp_yards, 0)
    }


def update_team_state(team_states, tid, result):
    for stat in STAT_NAMES:
        val = result[stat]
        team_states[tid][stat]["roll3_deque"].append(val)
        team_states[tid][stat]["roll8_deque"].append(val)
        team_states[tid][stat]["season_vals"].append(val)


def estimate_tds_from_score_v5(engine, team_score, tid):
    if team_score == 0:
        return 0, 0
    avg_tds = engine.team_td_rates.get(tid, engine.team_td_rates.mean())
    tds = np.random.poisson(avg_tds)
    tds = min(tds, team_score // 6)
    remaining = team_score - (tds * 7)
    while remaining < 0 and tds > 0:
        tds -= 1
        remaining = team_score - (tds * 7)
    return int(tds), int(max(remaining // 3, 0))


def sample_player_shares_v2(rank_shares_table, num_players, noise_std=0.35, bench_zero_prob=0.35, zero_eligible_ranks=None):
    base_shares = np.array([rank_shares_table.get(float(r), 0) for r in range(1, num_players + 1)])
    noisy_shares = base_shares * np.exp(np.random.normal(0, noise_std, size=num_players))
    zero_eligible_ranks = zero_eligible_ranks or []
    for i in range(num_players):
        if (i + 1) in zero_eligible_ranks and np.random.random() < bench_zero_prob:
            noisy_shares[i] = 0.001
    return noisy_shares / noisy_shares.sum()


def distribute_stats_capped_noisy_v2(team_total, current_depth_team, rank_shares_table, max_players,
                                      noise_std=0.35, zero_eligible_ranks=None, bench_zero_prob=0.35):
    eligible = current_depth_team[current_depth_team["pos_rank"] <= max_players].copy()
    n = len(eligible)
    if n == 0:
        return pd.DataFrame(columns=["player_name", "gsis_id", "rank", "share", "simulated_stat"])
    noisy_shares = sample_player_shares_v2(rank_shares_table, n, noise_std, bench_zero_prob, zero_eligible_ranks)
    result = []
    for i, (_, player) in enumerate(eligible.iterrows()):
        player_stat = team_total * noisy_shares[i]
        result.append({"player_name": player["player_name"], "gsis_id": player["gsis_id"],
                        "rank": player["pos_rank"], "share": noisy_shares[i], "simulated_stat": round(player_stat, 1)})
    return pd.DataFrame(result)


def distribute_receiving_capped_noisy_v2(team_total_rec_yards, current_wrs, current_tes, wr_rank_shares, te_rank_shares,
                                          position_split, max_wr=4, max_te=2, noise_std=0.35):
    wr_pool = team_total_rec_yards * position_split["WR"]
    te_pool = team_total_rec_yards * position_split["TE"]
    wr_results = distribute_stats_capped_noisy_v2(wr_pool, current_wrs, wr_rank_shares, max_wr, noise_std, zero_eligible_ranks=[4], bench_zero_prob=0.35)
    te_results = distribute_stats_capped_noisy_v2(te_pool, current_tes, te_rank_shares, max_te, noise_std, zero_eligible_ranks=[2], bench_zero_prob=0.35)
    return pd.concat([wr_results, te_results], ignore_index=True)


def get_playoff_seeds(season_group):
    results = []
    for conf in ["AFC", "NFC"]:
        conf_teams = season_group[season_group["team_conf"] == conf].copy()
        conf_teams = conf_teams.sort_values(["wins", "avg_points"], ascending=[False, False])
        division_winners = conf_teams.groupby("team_division").head(1).copy()
        division_winners = division_winners.sort_values(["wins", "avg_points"], ascending=[False, False])
        division_winners["seed"] = range(1, len(division_winners) + 1)
        remaining = conf_teams[~conf_teams["team_id"].isin(division_winners["team_id"])]
        wild_cards = remaining.sort_values(["wins", "avg_points"], ascending=[False, False]).head(3).copy()
        wild_cards["seed"] = range(len(division_winners) + 1, len(division_winners) + 1 + len(wild_cards))
        conf_playoff_teams = pd.concat([division_winners, wild_cards])
        conf_playoff_teams["conf"] = conf
        results.append(conf_playoff_teams)
    return pd.concat(results)


def break_playoff_tie(result):
    if result["points_for"] == result["points_against"]:
        ot_score = np.random.choice([3, 6, 7, 8], p=[0.55, 0.05, 0.30, 0.10])
        if np.random.random() < 0.5:
            result["points_for"] += ot_score
        else:
            result["points_against"] += ot_score
    return result


def _distribute_player_stats_for_team(engine, tid, team_yards, team_score):
    pass_share = engine.team_split["pass_share"].get(tid, engine.team_split["pass_share"].mean())
    team_pass_yards = team_yards * pass_share
    team_rush_yards = team_yards * (1 - pass_share)
    team_rec_yards = team_pass_yards

    qb_rows = distribute_stats_capped_noisy_v2(team_pass_yards, engine.current_qbs[engine.current_qbs["team_id"] == tid], engine.pass_shares, max_players=1, noise_std=0.2)
    rb_rows = distribute_stats_capped_noisy_v2(team_rush_yards, engine.current_rbs[engine.current_rbs["team_id"] == tid], engine.rush_shares, max_players=2, noise_std=0.35)
    rec_rows = distribute_receiving_capped_noisy_v2(team_rec_yards,
                                                     engine.current_wrs[engine.current_wrs["team_id"] == tid],
                                                     engine.current_tes[engine.current_tes["team_id"] == tid],
                                                     engine.wr_shares, engine.te_shares, engine.position_share, max_wr=4, max_te=2, noise_std=0.35)

    total_tds, total_fgs = estimate_tds_from_score_v5(engine, team_score, tid)
    rush_td_share = engine.team_td_counts["rush_td_share_correct"].get(tid, engine.team_td_counts["rush_td_share_correct"].mean())
    rush_tds_target = round(total_tds * rush_td_share)
    pass_tds_target = total_tds - rush_tds_target

    ypc = engine.rush_rates.loc[tid, "yards_per_carry"]
    ypt_catch = engine.rec_rates.loc[tid, "yards_per_target"] / engine.rec_rates.loc[tid, "catch_rate"]
    comp_pct = engine.pass_rates.loc[tid, "comp_pct"]

    rb_rows = rb_rows.copy()
    if len(rb_rows) > 0:
        rb_rows["carries"] = rb_rows["simulated_stat"].apply(lambda y: max(round(y / ypc), 1) if ypc > 0 else 1)
        rb_weights = (rb_rows["simulated_stat"] / rb_rows["simulated_stat"].sum()).values
        rb_rows["tds"] = np.random.multinomial(rush_tds_target, rb_weights) if rush_tds_target > 0 else [0] * len(rb_rows)
    else:
        rb_rows["carries"], rb_rows["tds"] = [], []

    rec_rows = rec_rows.copy()
    if len(rec_rows) > 0:
        rec_rows["receptions"] = rec_rows["simulated_stat"].apply(lambda y: max(round(y / ypt_catch), 1) if ypt_catch > 0 else 1)
        rec_weights = (rec_rows["simulated_stat"] / rec_rows["simulated_stat"].sum()).values
        rec_rows["tds"] = np.random.multinomial(pass_tds_target, rec_weights) if pass_tds_target > 0 else [0] * len(rec_rows)
        total_receptions = rec_rows["receptions"].sum()
    else:
        rec_rows["receptions"], rec_rows["tds"] = [], []
        total_receptions = 0

    qb_rows = qb_rows.copy()
    if len(qb_rows) > 0:
        qb_rows["completions"] = total_receptions
        qb_rows["attempts"] = max(round(total_receptions / comp_pct), total_receptions) if comp_pct > 0 else total_receptions
        qb_rows["tds"] = pass_tds_target

    return qb_rows, rb_rows, rec_rows


def simulate_one_season_full(engine, team_states_init, team_strength=None):
    team_states = copy.deepcopy(team_states_init)
    game_results, player_stat_rows = [], []

    for _, game in engine.schedule_2026.iterrows():
        home_id, away_id = game["home_id"], game["away_id"]
        is_indoor = 1 if game["roof"] in ["dome", "closed"] else 0
        temp = 70 if is_indoor else 60
        wind = 0 if is_indoor else 7

        sim_diff, _ = predict_game(
            engine, team_states, team_id=home_id, opp_id=away_id,
            is_home=1, div_game=game["div_game"], rest_days=7, opp_rest_days=7,
            temp=temp, wind=wind, is_indoor=is_indoor, qb_out=0, rb_out=0, wr_out=0,
            team_strength=team_strength
        )
        result = simulate_game_result(engine, team_states, home_id, away_id, sim_diff)

        game_results.append({
            "game_id": game["game_id"], "week": game["week"],
            "home_id": home_id, "away_id": away_id,
            "home_score": result["points_for"], "away_score": result["points_against"],
        })

        for tid, team_yards, team_score in [(home_id, result["total_yards"], result["points_for"]),
                                              (away_id, result["def_total_yards"], result["points_against"])]:
            qb_rows, rb_rows, rec_rows = _distribute_player_stats_for_team(engine, tid, team_yards, team_score)
            for df, stat_type in [(qb_rows, "passing"), (rb_rows, "rushing"), (rec_rows, "receiving")]:
                df = df.copy()
                df["stat_type"] = stat_type
                df["team_id"] = tid
                df["game_id"] = game["game_id"]
                df["week"] = game["week"]
                player_stat_rows.append(df)

        update_team_state(team_states, home_id, result)
        update_team_state(team_states, away_id, {
            "points_for": result["points_against"], "points_against": result["points_for"],
            "total_epa": result["def_total_epa"], "def_total_epa": result["total_epa"],
            "total_yards": result["def_total_yards"], "def_total_yards": result["total_yards"]
        })

    return team_states, pd.DataFrame(game_results), pd.concat(player_stat_rows, ignore_index=True)


def simulate_playoff_game_full(engine, team_states, team_a_id, team_b_id, round_name, neutral_site=False, team_strength=None):
    is_home_a = 0 if neutral_site else 1
    sim_diff, _ = predict_game(
        engine, team_states, team_id=team_a_id, opp_id=team_b_id,
        is_home=is_home_a, div_game=0, rest_days=7, opp_rest_days=7,
        temp=70, wind=0, is_indoor=1, qb_out=0, rb_out=0, wr_out=0,
        team_strength=team_strength
    )
    result = simulate_game_result(engine, team_states, team_a_id, team_b_id, sim_diff)
    result = break_playoff_tie(result)
    winner = team_a_id if result["points_for"] > result["points_against"] else team_b_id

    game_record = {
        "round": round_name, "team_a": team_a_id, "team_b": team_b_id,
        "score_a": result["points_for"], "score_b": result["points_against"], "winner": winner
    }

    player_rows = []
    for tid, team_yards, team_score in [(team_a_id, result["total_yards"], result["points_for"]),
                                          (team_b_id, result["def_total_yards"], result["points_against"])]:
        qb_rows, rb_rows, rec_rows = _distribute_player_stats_for_team(engine, tid, team_yards, team_score)
        for df, stat_type in [(qb_rows, "passing"), (rb_rows, "rushing"), (rec_rows, "receiving")]:
            df = df.copy()
            df["stat_type"] = stat_type
            df["team_id"] = tid
            df["round"] = round_name
            player_rows.append(df)

    update_team_state(team_states, team_a_id, result)
    update_team_state(team_states, team_b_id, {
        "points_for": result["points_against"], "points_against": result["points_for"],
        "total_epa": result["def_total_epa"], "def_total_epa": result["total_epa"],
        "total_yards": result["def_total_yards"], "def_total_yards": result["total_yards"]
    })
    return winner, game_record, pd.concat(player_rows, ignore_index=True)


def simulate_conference_bracket_full(engine, team_states, seeds_dict, conf_name, playoff_game_log, playoff_player_log, team_strength=None):
    r1 = simulate_playoff_game_full(engine, team_states, seeds_dict[2], seeds_dict[7], f"{conf_name} Wild Card", team_strength=team_strength)
    r2 = simulate_playoff_game_full(engine, team_states, seeds_dict[3], seeds_dict[6], f"{conf_name} Wild Card", team_strength=team_strength)
    r3 = simulate_playoff_game_full(engine, team_states, seeds_dict[4], seeds_dict[5], f"{conf_name} Wild Card", team_strength=team_strength)
    for winner, record, players in [r1, r2, r3]:
        playoff_game_log.append(record)
        playoff_player_log.append(players)

    wc_winners = [r1[0], r2[0], r3[0]]
    wc_seeds = {seeds_dict[2]: 2, seeds_dict[3]: 3, seeds_dict[4]: 4, seeds_dict[5]: 5, seeds_dict[6]: 6, seeds_dict[7]: 7}
    wc_winners_sorted = sorted(wc_winners, key=lambda t: wc_seeds[t])
    opponent_for_1 = wc_winners_sorted[-1]
    other_two = [t for t in wc_winners if t != opponent_for_1]

    d1 = simulate_playoff_game_full(engine, team_states, seeds_dict[1], opponent_for_1, f"{conf_name} Divisional", team_strength=team_strength)
    if wc_seeds[other_two[0]] < wc_seeds[other_two[1]]:
        d2 = simulate_playoff_game_full(engine, team_states, other_two[0], other_two[1], f"{conf_name} Divisional", team_strength=team_strength)
    else:
        d2 = simulate_playoff_game_full(engine, team_states, other_two[1], other_two[0], f"{conf_name} Divisional", team_strength=team_strength)
    for winner, record, players in [d1, d2]:
        playoff_game_log.append(record)
        playoff_player_log.append(players)

    cc = simulate_playoff_game_full(engine, team_states, d1[0], d2[0], f"{conf_name} Championship", team_strength=team_strength)
    playoff_game_log.append(cc[1])
    playoff_player_log.append(cc[2])
    return cc[0]


def simulate_full_playoffs(engine, team_states, this_season_seeds, team_strength=None):
    afc_seeds_dict = this_season_seeds[this_season_seeds["conf"] == "AFC"].set_index("seed")["team_id"].to_dict()
    nfc_seeds_dict = this_season_seeds[this_season_seeds["conf"] == "NFC"].set_index("seed")["team_id"].to_dict()

    playoff_game_log, playoff_player_log = [], []
    afc_champion = simulate_conference_bracket_full(engine, team_states, afc_seeds_dict, "AFC", playoff_game_log, playoff_player_log, team_strength=team_strength)
    nfc_champion = simulate_conference_bracket_full(engine, team_states, nfc_seeds_dict, "NFC", playoff_game_log, playoff_player_log, team_strength=team_strength)
    sb_winner, sb_record, sb_players = simulate_playoff_game_full(engine, team_states, afc_champion, nfc_champion, "Super Bowl", neutral_site=True, team_strength=team_strength)
    playoff_game_log.append(sb_record)
    playoff_player_log.append(sb_players)

    return pd.DataFrame(playoff_game_log), pd.concat(playoff_player_log, ignore_index=True), sb_winner, afc_champion, nfc_champion
