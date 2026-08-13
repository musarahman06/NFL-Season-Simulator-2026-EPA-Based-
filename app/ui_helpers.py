import streamlit as st
import pandas as pd
import io

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def darken_hex(hex_color, factor=0.55):
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return "#4d4d4d"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_team_colors(teams_full, team_id):
    """Primary color from the teams table + an auto-darkened secondary. Falls back to a
    neutral blue pair if none of the common color column names are found -- check your
    actual `teams` table column name and add it to CANDIDATES below if the fallback shows up."""
    CANDIDATES = ["team_color", "primary_color", "color", "team_color1"]
    row = teams_full[teams_full["team_id"] == team_id]
    primary = None
    if not row.empty:
        for col in CANDIDATES:
            if col in teams_full.columns:
                val = row.iloc[0].get(col)
                if pd.notna(val) and str(val).strip():
                    primary = str(val).strip()
                    break
    if primary is None:
        primary = "#1f77b4"
    elif not primary.startswith("#"):
        primary = f"#{primary}"
    return primary, darken_hex(primary)


def get_team_logo(teams_full, team_id):
    """Returns a logo URL if the teams table has one under a common column name, else None
    (rendering code should handle None gracefully -- not every project's teams table has logos)."""
    CANDIDATES = ["team_logo_espn", "team_logo_wikipedia", "team_logo", "logo_url", "team_wordmark"]
    row = teams_full[teams_full["team_id"] == team_id]
    if row.empty:
        return None
    for col in CANDIDATES:
        if col in teams_full.columns:
            val = row.iloc[0].get(col)
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return None


def render_position_groups(team_stats):
    """Renders passing/rushing/receiving lines for one team in one game, labeled by position group."""
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


def render_playoff_bracket(games_df, players_df, team_abbr_lookup, teams_full, seed_lookup=None):
    """Renders a full playoff bracket (AFC/NFC Wild Card -> Divisional -> Championship,
    then the Super Bowl) from a playoff games/players table -- works identically whether
    that table came from the saved single-season sim or a freshly-run custom sim, since
    both share the same column structure.

    seed_lookup: optional {team_id: seed_number} dict -- if provided, seeds are shown
    next to each team's abbreviation (e.g. "(#3) KC")."""

    def abbr(tid):
        base = team_abbr_lookup.get(tid, "UNK")
        if seed_lookup and tid in seed_lookup and pd.notna(seed_lookup[tid]):
            return f"(#{int(seed_lookup[tid])}) {base}"
        return base

    sb_row_df = games_df[games_df["round"] == "Super Bowl"]
    if not sb_row_df.empty:
        sb_row = sb_row_df.iloc[0]
        champ_id = sb_row["winner"]
        st.success(f"🏆 **Super Bowl Champion: {abbr(champ_id)}**")
        champ_logo = get_team_logo(teams_full, champ_id)
        if champ_logo:
            st.image(champ_logo, width=100)

    round_map = {
        "AFC": ["AFC Wild Card", "AFC Divisional", "AFC Championship"],
        "NFC": ["NFC Wild Card", "NFC Divisional", "NFC Championship"],
    }

    afc_col, nfc_col = st.columns(2)
    for col, conf in [(afc_col, "AFC"), (nfc_col, "NFC")]:
        with col:
            st.subheader(conf)
            for rnd in round_map[conf]:
                round_games = games_df[games_df["round"] == rnd]
                if round_games.empty:
                    continue
                st.markdown(f"**{rnd.replace(conf + ' ', '')}**")
                for _, g in round_games.iterrows():
                    a_abbr, b_abbr, w_abbr = abbr(g["team_a"]), abbr(g["team_b"]), abbr(g["winner"])
                    label = f"{a_abbr} {g['score_a']:.0f} - {g['score_b']:.0f} {b_abbr}  (W: {w_abbr})"
                    with st.expander(label):
                        col_a, col_b = st.columns(2)
                        for c, tid, tabbr in [(col_a, g["team_a"], a_abbr), (col_b, g["team_b"], b_abbr)]:
                            with c:
                                logo = get_team_logo(teams_full, tid)
                                if logo:
                                    st.image(logo, width=50)
                                st.markdown(f"**{tabbr}**")
                                team_stats = players_df[(players_df["round"] == g["round"]) & (players_df["team_id"] == tid)]
                                team_stats = team_stats[team_stats["simulated_stat"] >= 1]
                                render_position_groups(team_stats)

    if not sb_row_df.empty:
        st.subheader("Super Bowl")
        sb_row = sb_row_df.iloc[0]
        a_abbr, b_abbr = abbr(sb_row["team_a"]), abbr(sb_row["team_b"])
        label = f"{a_abbr} {sb_row['score_a']:.0f} - {sb_row['score_b']:.0f} {b_abbr}"
        with st.expander(label, expanded=True):
            col_a, col_b = st.columns(2)
            for c, tid, tabbr in [(col_a, sb_row["team_a"], a_abbr), (col_b, sb_row["team_b"], b_abbr)]:
                with c:
                    logo = get_team_logo(teams_full, tid)
                    if logo:
                        st.image(logo, width=50)
                    st.markdown(f"**{tabbr}**")
                    team_stats = players_df[(players_df["round"] == "Super Bowl") & (players_df["team_id"] == tid)]
                    team_stats = team_stats[team_stats["simulated_stat"] >= 1]
                    render_position_groups(team_stats)


def render_division_standings(record_df, team_abbr_lookup, teams_full, seed_lookup=None):
    """Renders full-season standings split by conference then division, with team logos,
    names, records, and playoff seed numbers where applicable.

    record_df must have columns: team_id, wins, losses, ties, avg_points, team_conf, team_division.
    seed_lookup: optional {team_id: seed_number} dict -- unseeded teams show no seed marker."""
    df = record_df.copy()
    df["Record"] = df["wins"].astype(str) + "-" + df["losses"].astype(str) + "-" + df["ties"].astype(str)
    df["Team"] = df["team_id"].map(team_abbr_lookup)

    for conf in ["AFC", "NFC"]:
        st.subheader(conf)
        conf_data = df[df["team_conf"] == conf]
        divisions = sorted(conf_data["team_division"].unique())
        cols = st.columns(len(divisions))
        for col, div in zip(cols, divisions):
            with col:
                st.markdown(f"**{div}**")
                div_data = conf_data[conf_data["team_division"] == div].sort_values(
                    ["wins", "avg_points"], ascending=[False, False]
                )
                for _, row in div_data.iterrows():
                    tid = row["team_id"]
                    seed = seed_lookup.get(tid) if seed_lookup else None
                    seed_label = f"#{int(seed)} " if seed is not None and pd.notna(seed) else ""

                    logo_col, info_col = st.columns([1, 3])
                    with logo_col:
                        logo = get_team_logo(teams_full, tid)
                        if logo:
                            st.image(logo, width=32)
                    with info_col:
                        st.markdown(f"**{seed_label}{row['Team']}**")
                        st.caption(f"{row['Record']}  •  {row['avg_points']:.1f} pts/g")


def clean_player_stats_csv(players_df, team_abbr_lookup, time_col="week", time_label="Week"):
    """Turns the raw simulation output (gsis_id, rank, share, mixed NaN columns depending
    on position) into a clean, human-readable table: team abbreviations instead of IDs,
    whole-number stats, readable column names, and only the columns that actually apply
    per position (blank rather than a stray 0.0 for stats that don't apply to a row)."""
    df = players_df.copy()
    df["Team"] = df["team_id"].map(team_abbr_lookup)
    df["Position"] = df["stat_type"].str.capitalize()
    df["Player"] = df["player_name"]
    df["Player ID"] = df["gsis_id"] if "gsis_id" in df.columns else pd.NA
    df["Yards"] = df["simulated_stat"].round(0).astype(int)
    df["TDs"] = df["tds"].round(0).astype(int) if "tds" in df.columns else 0

    for col in ["completions", "attempts", "carries", "receptions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0)
        else:
            df[col] = pd.NA

    df = df.rename(columns={
        "completions": "Completions", "attempts": "Attempts",
        "carries": "Carries", "receptions": "Receptions",
    })

    out_cols = []
    if time_col in df.columns:
        df = df.rename(columns={time_col: time_label})
        out_cols.append(time_label)
    out_cols += ["Team", "Player", "Player ID", "Position", "Yards", "TDs", "Completions", "Attempts", "Carries", "Receptions"]
    out = df[out_cols].copy()

    for col in ["Completions", "Attempts", "Carries", "Receptions"]:
        out[col] = out[col].astype("Int64")  # nullable int -- leaves a real blank instead of "0.0" where a stat doesn't apply

    if time_label in out.columns:
        out = out.sort_values([time_label, "Team", "Yards"], ascending=[True, True, False])
    else:
        out = out.sort_values(["Team", "Yards"], ascending=[True, False])

    return out.reset_index(drop=True)


def _fetch_logo_flowable(url, size=18):
    """Downloads a logo image and wraps it as a reportlab Image flowable for use inside a
    Table cell. Returns an empty string (blank cell) if the download fails for any reason --
    a missing logo should never break PDF generation."""
    if not url:
        return ""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = resp.read()
        return Image(io.BytesIO(data), width=size, height=size)
    except Exception:
        return ""


def generate_standings_pdf(record_df, team_abbr_lookup, teams_full, seed_lookup=None,
                            power_rank_lookup=None, epa_lookup=None, title="Simulated Season Standings"):
    """Builds a PDF of the division-standings breakdown (same grouping as render_division_standings),
    with team logos and optional Power Rank / EPA columns, and returns it as bytes ready for
    st.download_button. Requires the `reportlab` package.

    power_rank_lookup: optional {team_id: rank} -- shown as a column if provided.
    epa_lookup: optional {team_id: starting offensive EPA} -- shown as a column if provided.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF export requires the 'reportlab' package. Install it with: pip install reportlab")

    df = record_df.copy()
    df["Record"] = df["wins"].astype(str) + "-" + df["losses"].astype(str) + "-" + df["ties"].astype(str)
    df["Team"] = df["team_id"].map(team_abbr_lookup)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"])]
    elements.append(Paragraph(
        "Power rankings based on NFL.com's preseason power rankings (published August 2026), "
        "adjusted per-simulation where noted.", styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    header = ["Logo", "Seed", "Team", "Record", "Avg Pts"]
    if power_rank_lookup:
        header.append("Power Rank")
    if epa_lookup:
        header.append("Off EPA")

    col_widths = [0.4 * inch, 0.5 * inch, 0.7 * inch, 0.85 * inch, 0.65 * inch]
    if power_rank_lookup:
        col_widths.append(0.7 * inch)
    if epa_lookup:
        col_widths.append(0.6 * inch)

    for conf in ["AFC", "NFC"]:
        elements.append(Paragraph(conf, styles["Heading2"]))
        conf_data = df[df["team_conf"] == conf]
        for div in sorted(conf_data["team_division"].unique()):
            elements.append(Paragraph(div, styles["Heading3"]))
            div_data = conf_data[conf_data["team_division"] == div].sort_values(
                ["wins", "avg_points"], ascending=[False, False]
            )

            table_data = [header]
            for _, row in div_data.iterrows():
                tid = row["team_id"]
                seed = seed_lookup.get(tid) if seed_lookup else None
                seed_str = f"#{int(seed)}" if seed is not None and pd.notna(seed) else "—"
                logo_cell = _fetch_logo_flowable(get_team_logo(teams_full, tid))

                row_data = [logo_cell, seed_str, row["Team"], row["Record"], f"{row['avg_points']:.1f}"]
                if power_rank_lookup:
                    pr = power_rank_lookup.get(tid)
                    row_data.append(f"#{int(pr)}" if pr is not None and pd.notna(pr) else "—")
                if epa_lookup:
                    epa_val = epa_lookup.get(tid)
                    row_data.append(f"{epa_val:+.2f}" if epa_val is not None and pd.notna(epa_val) else "—")
                table_data.append(row_data)

            t = Table(table_data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 10))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
