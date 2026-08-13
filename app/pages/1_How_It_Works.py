import streamlit as st

st.set_page_config(page_title="How It Works", page_icon="📖", layout="wide")

st.title("📖 How This Simulator Works")
st.caption("A plain-English glossary for the stats and methods used throughout the app.")

with st.expander("**EPA (Expected Points Added)**", expanded=True):
    st.markdown("""
    EPA measures how much a single play changed a team's expected points on the scoreboard —
    not the points scored on that play, but how much *more* (or less) likely the team was to
    score, based on down, distance, field position, and time remaining.

    A 5-yard run on 3rd-and-15 barely moves the needle (low EPA). A 5-yard run on 3rd-and-4 that
    gets a first down is worth a lot more (higher EPA), even though the yardage is identical.
    This is why EPA is a better measure of *how good a team actually is* than raw yards or points —
    it accounts for context.

    `total_epa` in this project is the sum of a team's passing, rushing, and receiving EPA in a
    given game. `def_total_epa` is the same idea applied to the opposing offense — effectively,
    how much EPA a team's defense *allowed*.
    """)

with st.expander("**Rolling averages (roll3 / roll8 / season)**"):
    st.markdown("""
    Rather than using a team's full-season average all year (which reacts slowly to a team getting
    hot or cold), the model tracks three separate windows for every stat:

    - **roll3** — average over the team's last 3 games. Reacts fastest, most sensitive to a recent
      hot or cold streak.
    - **roll8** — average over the team's last 8 games. A steadier read on current form.
    - **season** — average over the current season only, resets to empty at the start of each new
      season.

    All three feed into the prediction model, so it can weigh "how has this team looked lately"
    against "how have they looked all year" when forecasting the next game.
    """)

with st.expander("**Adjusted EPA**"):
    st.markdown("""
    Raw EPA doesn't account for opponent strength — a team can look great on offense just because
    they've played bad defenses. Adjusted EPA corrects for this:

    ```
    adjusted_offensive_epa = team's own rolling offensive EPA − opponent's rolling defensive EPA
    ```

    This gives a cleaner signal of "how good is this offense, relative to what a league-average
    defense would have allowed" — and the same logic applies in reverse for adjusted defensive EPA.
    """)

with st.expander("**The prediction model**"):
    st.markdown("""
    An XGBoost model, trained on years of real historical games, takes in each team's rolling and
    adjusted stats (plus situational factors like rest days, weather, and injuries) and predicts
    the point differential for a given matchup. Random noise, calibrated to the model's real-world
    error rate, is layered on top so no single simulated game is ever treated as a certainty —
    it's a distribution of plausible outcomes, not one fixed answer.
    """)

with st.expander("**Depth charts & injuries**"):
    st.markdown("""
    Two separate ways real roster information feeds into this project:

    - **As a prediction input** — whether a team's starting QB, RB, or WR is listed as injured
      (`qb_out`, `rb_starter_out`, `wr_starter_out`) is one of the features the model uses to
      predict a game's outcome. This is scoped to those three offensive skill positions since
      injury-report data for other positions was too inconsistent to trust reliably.
    - **For simulated box scores** — team-level stats (like total rushing yards in a simulated
      game) get distributed down to individual players using each team's **current real depth
      chart**, weighted by how much production a player at that depth-chart rank has historically
      gotten (a team's RB1 gets a bigger share than their RB3, for example). This is what lets the
      app show realistic player-level stat lines for a simulated game, using the actual players on
      each team's current roster.
    """)

with st.expander("**Preseason power rankings nudge**"):
    st.markdown("""
    Before real 2026 results exist, the model has little to go on for how good a team actually is.
    To fill that gap, NFL.com's preseason power rankings (1–32) contribute two small, decaying
    effects — never a guarantee of anything:

    - A **temporary nudge** early in the season, which fades out entirely by around a team's 9th
      game, once real simulated results have taken over.
    - A **small permanent pull** that never fully disappears, reflecting a modest, standing
      talent-tier effect.

    Both are small relative to the model's normal game-to-game randomness — a #1-ranked team can
    still lose, and a #32-ranked team can still go on a surprise run.
    """)

with st.expander("**Monte Carlo simulation**"):
    st.markdown("""
    Any single simulated season is just one possible outcome — a lot of randomness goes into every
    game. To get a reliable sense of each team's *true* odds, the whole 2026 season (every game, in
    order) gets simulated 500 separate times, each one independent of the others. A team's
    "playoff %" is simply the share of those 500 simulated seasons in which they made the playoffs
    — same idea for Super Bowl odds.

    Separately, this app also shows results from **one specific detailed season** (with full player
    box scores) — that single run is what powers the Team Pages, Box Score Lookup, and Playoff
    Bracket pages, since generating player-level detail for all 500 seasons would be far too slow.
    """)

st.divider()
st.caption("Built with a trained XGBoost point-differential model, real historical NFL play-by-play data, and Monte Carlo simulation.")
