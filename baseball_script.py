'''
Script to use when drafting players. Updates stats as players are drafted to determine optimal picks based on other team's draft choices. 
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

player_df = pd.read_csv('player_data.csv')
hitting_df = pd.read_csv('hitting_data.csv')
pitching_df = pd.read_csv('pitching_data.csv')

player_df.set_index('Player #', inplace=True)
# convert column 'Player' to numeric, dropping commas 
player_df['Player'] = player_df['Player'].str.replace(',', '').astype(float)


pitching_df.set_index('TEAM', inplace=True)
# drop duplicate columns
pitching_df.drop('W', axis=1, inplace=True)
pitching_df.drop('L', axis=1, inplace=True)
pitching_df.drop('GP', axis=1, inplace=True)
pitching_df.drop('BB', axis=1, inplace=True)
pitching_df.drop('R', axis=1, inplace=True)



hitting_df.set_index('TEAM', inplace=True)
hitting_df.rename(columns={'R':'R_hitting'}, inplace=True)

# merge hitting and pitching data
team_df = pd.merge(hitting_df, pitching_df, left_index=True, right_index=True)
team_df.drop('Team #', axis=1, inplace=True)

# determine which columns are present in both dataframes, these can be used as features
keep_cols = set(player_df.columns).intersection(set(team_df.columns))
keep_cols.add('TEAM')
keep_cols.add('R_hitting')
keep_cols.add('R_pitching')


col_descriptions = {
    'GP' : 'Games Played',
    'AB' : 'At Bats',
    'R_hitting' : 'Runs Scored',
    'H' : 'Hits',
    '2B' : 'Doubles',
    '3B' : 'Triples',
    'HR' : 'Home Runs',
    'TB' : 'Total Bases',
    'RBI' : 'Runs Batted In',
    'BA' : 'Batting Average',
    'OBP' : 'On Base Percentage',
    'SLG' : 'Slugging Percentage',
    'OPS' : 'On Base Plus Slugging',
    'W' : 'Wins',
    'L' : 'Losses',
    'ERA' : 'Earned Run Average',
    'SV' : 'Saves',
    'CG' : 'Complete Games',
    'SHO' : 'Shutouts',
    'IP' : 'Innings Pitched',
    'QS' : 'Quality Starts',
    'ER' : 'Earned Runs',
    'R_pitching' : 'Runs Allowed',
    'BB' : 'Walks',
    'SO' : 'Strikeouts',
    'BAA' : 'Opponent Batting Average',
    'R' : 'Runs',
    'SB' : 'Stolen Bases',
    'CS' : 'Caught Stealing'}


def getSigCols(corr_tolerance, df):
    # eliminating variables with low correlation 
    significant_cols = []
    for i in df.columns:
        correlation = df['W'].corr(df[i])
        if abs(correlation) < corr_tolerance or i == 'W' or i == 'L' or i == 'GP' or i not in keep_cols:
            continue
        significant_cols.append((i,correlation))
    return significant_cols

def get_top_players(team, available_players, significant_cols, current_team, player_df, team_df, n=10, budget=30000000, picks_remaining=4):
    # player_df is a scaled [0,1] dataframe of player stats, aside from 'Player Cost' and 'Player ID'
    # get the team's current stats
    league_avg = team_df.loc['LA Angels'] # use team with most wins 
    player_scores = {}

    # update column weights based on team's current stats
    significant_cols_delta = {}
    ##########################################################################
    # Draft Step 4, Part 1: calculating delta
    for column, weight in significant_cols:
        team_stat = current_team[column]
        league_stat = league_avg[column]
        diff = team_stat - league_stat
        if weight > 0:
            diff = diff * -1
        # divide diff by the league average
        diff = diff / league_avg[column]
        significant_cols_delta[column] = diff
    # print('Significant Columns delta from league average: ', significant_cols_delta)
    
    ##########################################################################
    # Draft Step 4, Part 2: calculating player scores
    for player_number, player_row in player_df.iterrows():
        player_number += 1
        if player_number not in available_players:
            continue
        # calculate the weighted average of the significant columns
        player_score = 0
        # print('Player: ', player_row)
        # convert to int, remove commas
        player_cost = player_row['Player Cost']
        if player_cost > budget:
            player_score = float('-inf')
        
        # print('calculating scores for player: ', player_row['Player ID'])
        # need 4 picks
        # budget originally 30M
        # average budget is 30M/4
        budget_score = 0
        # budget score is how close the player_cost is to the budget/picks_remaining

        if player_cost < budget:
            target_budget = budget/picks_remaining
            budget_score = -1* abs(player_cost - target_budget)/target_budget 


        for col, weight in significant_cols:
            weight *= 100 
            weight *= significant_cols_delta[col]
            # multiply significant column by the column weight
            if col == 'R_hitting':
                # player_row['R'] = the player's hitting runs
                # print('incrementing hitting score by ', player_row['R'] * weight)
                player_score += player_row['R'] * weight
            elif col == 'R_pitching':
                # print('incrementing pitching score by (negative)', player_row['R'] * weight)
                player_score -= player_row['R'] * weight
            
            else:
                if weight > 0:
                    player_score += player_row[col] * weight

                else:
                    player_score -= player_row[col] * weight
        # summarize scores
        # print(f'Player Score (player {player_number}): ', player_score)
        # print(f'Player Budget Score (player {player_number}): ', budget_score)
        player_scores[player_number] = player_score + budget_score # don't care about budget for now
    ##########################################################################
    # Draft Step 5: return the top n players
    # sort the player scores, and return the top n players
    sorted_player_scores = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    # print('Sorted Player Scores: ', sorted_player_scores)
    return sorted_player_scores[:n]


def draft(team='Seattle', corr_level=0.25, n=10, budget=30000000):
    '''
    team: team to draft for
    corr_level: correlation level to use for determining significant columns
    n: number of players to draft
    budget: budget to use for drafting
    '''
    current_team = team_df.loc[team]
    # print(current_team)
    # get significant columns
    significant_cols = getSigCols(corr_level, team_df)
    print(f'Significant Columns: ', significant_cols)


    available_players = list(player_df.index)
    current_team = team_df.loc[team]

    cur_player_df = player_df.copy()
    cur_player_df['Player ID'] = cur_player_df.index
    sigcols = getSigCols(corr_level)
    
    picks_remaining = 4

    normalized_cur_df = cur_player_df.drop('Player', axis=1)
    normalized_cur_df = scaler.fit_transform(normalized_cur_df)
    normalized_cur_df = pd.DataFrame(normalized_cur_df, columns=cur_player_df.columns[1:])
    normalized_cur_df['Player ID'] = cur_player_df['Player ID']
    normalized_cur_df['Player Cost'] = cur_player_df['Player']
    


    while True:
        # prompt user for input
        print(f'Enter 1 to draft a player for your team ({team}) or 2 to draft a player for another team (enter nothing to quit)')
        choice = input()
        if choice == '1':
            # draft for our team
            print('Available Players: ', available_players)
            # get the top n players
            top_players = get_top_players(team, available_players, sigcols, current_team=current_team, n=n, budget=budget, player_df=normalized_cur_df, team_df=team_df, picks_remaining=picks_remaining) 
            # print(top_players)
            print('\n')
            print('Current Budget: {:,.0f}'.format(budget))
            print('\n')

            print(f'Top {n} Players: ')
            for i, player in enumerate(top_players):
                print(f'Rank {i+1} --> Player: {player[0]}, Score: {np.round(player[1],2)}', 'Cost: {:,.0f}'.format(normalized_cur_df.loc[player[0], 'Player Cost']))

            print('Enter the Player # (not rank) to draft or enter nothing to quit')
            player_num = int(input())

            if player_num == '':
                print('Are you sure you want to quit? (y/n)')
                choice = input()
                if choice == 'y':
                    break
                else: 
                    continue
            if player_num not in available_players:
                print('Player not available')
                continue

            # draft the player
            # find the player in the player_df
            player = player_num
            
            # player = top_players[int(player_num)]
            print(f'Drafting {player} for {team}')
            # current_team = current_team + player_df.loc[player]
            for col in player_df:
                if col == 'R':
                    current_team['R_hitting'] += player_df.loc[player, col]
                    # current_team['R_pitching'] -= player_df.loc[player, col]/2

                if col in current_team:
                    current_team[col] += player_df.loc[player, col]
            # print(current_team, player_df.loc[player])
            # remove the player from the available players
            available_players.remove(player)
            picks_remaining -= 1
        elif choice == '2':
            # register a player for another team
            # remove the player from the available players
            cont = True
            while cont:
                print('What player was drafted?')
                choice = int(input())
                print('Enter 1 to confirm or 2 to re-enter')
                confirm = input()
                if confirm == '1':
                    available_players.remove(choice)
                    cont = False
                else:
                    continue
        elif choice == '':
            # quit
            print('Are you sure you want to quit? (y/n)')
            choice = input()
            if choice == 'y':
                break
            else: 
                continue
        else:
            print('Invalid choice, try again.')
            continue

draft('Seattle', 0.2)