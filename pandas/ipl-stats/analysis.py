"""
IPL Stats Analyzer
------------------


Data: data/matches.csv, data/deliveries.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Quick sanity check — always do this first with a new dataset
print("Matches Columns:", matches.columns.tolist())
print("Deliveries Columns:", deliveries.columns.tolist())
print(matches.head(2))
print(deliveries.head(2))


# ----------------------------------------------------------------
# Q1: Which teams have won the most matches?
# ----------------------------------------------------------------
def most_successful_teams():
    """
    Returns a Series: team -> win count, sorted descending.
    """
    return matches['winner'].value_counts()


# ----------------------------------------------------------------
# Q2: Strike rate by phase (powerplay 1-6, middle 7-15, death 16-20)
# ----------------------------------------------------------------
def strike_rate_by_phase():
    """
    Calculates strike rate by phase for batters with >= 30 balls faced in that phase.
    """
    df = deliveries.copy()

    # Define match phases based on over number (1-6, 7-15, 16-20)
    bins = [0, 6, 15, 20]
    labels = ['powerplay', 'middle', 'death']
    df['phase'] = pd.cut(df['over'], bins=bins, labels=labels)

    # Group by batter and phase
    grouped = df.groupby(['batter', 'phase'], as_index=False).agg(
        total_runs=('batsman_runs', 'sum'),
        balls_faced=('ball', 'count')
    )

    # Filter for sample size robustness
    filtered_df = grouped[grouped['balls_faced'] >= 30].copy()

    # Calculate Strike Rate
    filtered_df['strike_rate'] = (filtered_df['total_runs'] / filtered_df['balls_faced']) * 100

    return filtered_df.sort_values(by='strike_rate', ascending=False)


# ----------------------------------------------------------------
# Q3: Most consistent bowler (lowest variance in economy rate across matches)
# ----------------------------------------------------------------
def most_consistent_bowler():
    """
    Finds bowlers with lowest standard deviation of economy rate across matches.
    Filters for bowlers with >= 10 matches.
    """
    df = deliveries.copy()

    # A ball is illegal if extras_type is 'wides' or 'noballs'
    df['is_legal_ball'] = np.where(df['extras_type'].isin(['wides', 'noballs']), 0, 1)

    # Bowler runs = batsman_runs + extra_runs (EXCEPT byes and legbyes, which don't count against bowler)
    df['is_byes_or_legbyes'] = df['extras_type'].isin(['byes', 'legbyes'])
    df['bowler_runs'] = np.where(df['is_byes_or_legbyes'], df['batsman_runs'], df['total_runs'])

    # Group by match and bowler
    match_bowler = df.groupby(['match_id', 'bowler']).agg(
        runs_conceded=('bowler_runs', 'sum'),
        legal_balls=('is_legal_ball', 'sum')
    ).reset_index()

    # Avoid division by zero, filter out entries with 0 legal balls
    match_bowler = match_bowler[match_bowler['legal_balls'] > 0]

    # Calculate economy rate for that match
    match_bowler['overs'] = match_bowler['legal_balls'] / 6
    match_bowler['economy'] = match_bowler['runs_conceded'] / match_bowler['overs']

    # Group by bowler to get consistency metrics
    bowler_stats = match_bowler.groupby('bowler').agg(
        match_count=('match_id', 'count'),
        economy_std=('economy', 'std')
    ).reset_index()

    # Filter for minimum 10 matches and sort by lowest standard deviation
    consistent_bowlers = bowler_stats[bowler_stats['match_count'] >= 10]
    return consistent_bowlers.sort_values(by='economy_std', ascending=True)


# ----------------------------------------------------------------
# Q4: Win probability when chasing vs defending
# ----------------------------------------------------------------
def chase_vs_defend_win_rate():
    """
    Computes: (number of matches won by chasing team) / (total matches with a clear winner)
    """
    valid_matches = matches[matches['winner'].notna()].copy()

    # Case 1: Toss winner chose to field and won the match
    chase_win_1 = (valid_matches['toss_decision'] == 'field') & (
                valid_matches['toss_winner'] == valid_matches['winner'])

    # Case 2: Toss winner chose to bat and lost the match (meaning the chasing team won)
    chase_win_2 = (valid_matches['toss_decision'] == 'bat') & (valid_matches['toss_winner'] != valid_matches['winner'])

    chasing_team_won = chase_win_1 | chase_win_2

    chase_win_rate = np.mean(chasing_team_won)
    defend_win_rate = 1 - chase_win_rate

    return {"Chase Win Rate": chase_win_rate, "Defend Win Rate": defend_win_rate}


# ----------------------------------------------------------------
# Q5: Best death-over (16-20) batsmen by strike rate
# ----------------------------------------------------------------
def best_death_over_batsmen():
    """
    Filters for death phase and computes top batsmen by strike rate.
    """
    df = deliveries.copy()

    # Filter for death overs (16 to 20)
    death_df = df[df['over'] >= 16].copy()

    # Group by batter
    grouped = death_df.groupby('batter').agg(
        total_runs=('batsman_runs', 'sum'),
        balls_faced=('ball', 'count')
    ).reset_index()

    # Filter for sample size (minimum 30 balls faced in death overs)
    filtered_df = grouped[grouped['balls_faced'] >= 30].copy()
    filtered_df['strike_rate'] = (filtered_df['total_runs'] / filtered_df['balls_faced']) * 100

    return filtered_df.sort_values(by='strike_rate', ascending=False)


# ----------------------------------------------------------------
# Q6: Does winning the toss correlate with winning the match?
# ----------------------------------------------------------------
def toss_match_correlation():
    """
    Encodes True/False as 1/0 and returns the mean probability.
    """
    valid_matches = matches[matches['winner'].notna()].copy()
    toss_and_match_win = (valid_matches['toss_winner'] == valid_matches['winner']).astype(int)

    return np.mean(toss_and_match_win)


# ----------------------------------------------------------------
# Q7: Custom "Impact Score" — your own formula, vectorized (no loops!)
# ----------------------------------------------------------------
def impact_score():
    """
    Formula: Impact = Runs Scored + (20 * Wickets Taken)
    Vectorized calculation per player-match, aggregated overall.
    """
    # 1. Get runs scored per (match_id, batter)
    batting_perf = deliveries.groupby(['match_id', 'batter'])['batsman_runs'].sum().reset_index()
    batting_perf.rename(columns={'batter': 'player', 'batsman_runs': 'runs'}, inplace=True)

    # 2. Get bowler credited wickets per (match_id, bowler)
    bowler_wickets = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    wicket_df = deliveries[deliveries['dismissal_kind'].isin(bowler_wickets)].copy()

    bowling_perf = wicket_df.groupby(['match_id', 'bowler'])['dismissal_kind'].count().reset_index()
    bowling_perf.rename(columns={'bowler': 'player', 'dismissal_kind': 'wickets'}, inplace=True)

    # 3. Outer merge to capture pure batters, pure bowlers, and all-rounders
    performance = pd.merge(batting_perf, bowling_perf, on=['match_id', 'player'], how='outer').fillna(0)

    # 4. Compute impact using vectorized arithmetic
    performance['match_impact'] = performance['runs'] + (20 * performance['wickets'])

    # 5. Group by player, sum across matches, sort descending
    total_impact = performance.groupby('player')['match_impact'].sum().reset_index()
    return total_impact.sort_values(by='match_impact', ascending=False)


if __name__ == "__main__":
    print("\n--- Q1: Most successful teams ---")
    print(most_successful_teams().head())

    print("\n--- Q2: Strike rate by phase ---")
    print(strike_rate_by_phase().head())

    print("\n--- Q3: Most consistent bowler ---")
    print(most_consistent_bowler().head())

    print("\n--- Q4: Chase vs defend win rate ---")
    print(chase_vs_defend_win_rate())

    print("\n--- Q5: Best death over batsmen ---")
    print(best_death_over_batsmen().head())

    print("\n--- Q6: Toss-match correlation ---")
    print(f"Toss winner wins match {toss_match_correlation() * 100:.2f}% of the time.")

    print("\n--- Q7: Impact score ---")
    print(impact_score().head())