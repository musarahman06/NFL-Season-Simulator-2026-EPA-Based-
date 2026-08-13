## 🏈 2026 NFL Season Simulator

This is a full **NFL season prediction** and sim engine built **end to end**. It is trained on XGBoost model that predicts every 2026 game and a Monte Carlo engine that runs the season 500 times to compute realistic playoff and "Big Game" odds. 

![Home page standings](screenshots/1MonteCarloSIm.png)

Includes a interactive Streamlit app built in collaboration with Claude to allow you to run your own sims with downloadable .csv's.

## What This Is

Given real historical NFL data from 2015-2024 pulled via nfl_data_py and each teams current depth chart, this project:
1. Predicts the point differentical for every 2026 regular season and playoff game using a trained **XGBoost Model**
2. Runs the full season 500 times (Monte Carlo simulation) to produce stable and realtics playoff and "Big Game" probabilities for all 32 teams
3. Runs one fully detailed season with real player level box scores
4. Wraps all of it into a interactive Steamlit app - including a page where you can adjust team strenghth inputs and power rankings to run your own sim

## How it works

- **Feature engineering**: rolling 3-game, 8-game, season-to-date team stats, opponent-adjusted EPA, rest days, weather and starter injury flags
- **Preason power rankings**: NFL.coms 2026 preson rankings feed a small, decaying Week nudge plus a permenent smallboost to a teams EPA throughout the season. This makes it so the #1 Team can't have a massive fall from grace.
- **Player stats**: team level yardage gets distrbuted down into player stats based on teams historical splits. For example, a team like the Cheifs rely heavily on their TE and WR1 so that is reflected
- **Monte Carlo**: 500 full season sims each with fresh randomness to make accurate playoff and "Big Game" odds

A handful of real bugs were caught and fixed along the way. Some include drive-count rounding bug that defalted league wid scoering across a season, massive discrepencies in QB attemps compared to real life and a playoff braket that had games ending in ties. The in-app How It Works page documents the methodology more in depth.
