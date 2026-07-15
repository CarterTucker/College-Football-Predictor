import pandas as pd
import os
import time
# import cfb_predictor_lib as cfb

df = pd.read_csv('games.csv')

teamInfo = {}

for _, game in df.iterrows():

    teamId = game['homeId']
    teamName = game['homeTeam']

    if teamId not in teamInfo.keys():
        teamInfo[teamId] = teamName

def get_team_name(teamId):
    if teamInfo:
        return teamInfo[teamId]
    else:
        return "teamInfo has not been initiated!"
    
def num_teams():
    return len(teamInfo.keys())

teamStrength = {}

# initiate team ratings
for teamId in teamInfo.keys():
    teamStrength[teamId] = 0

# define a learning rate
beta = 0.03

counter = 0
total_games = 0
success = 0
for _, game in df.iterrows():
    try:
        homeId = game['homeId']
        awayId = game['awayId']

        expectedMargin = teamStrength[homeId] - teamStrength[awayId]
        error = game['margin'] - expectedMargin

        teamStrength[homeId] += beta*error
        teamStrength[awayId] -= beta*error

        total_games += 1
        if (expectedMargin > 0 and game['margin'] > 0) or (expectedMargin < 0 and game['margin'] < 0) or (expectedMargin == 0 and game['margin'] == 0):
            success += 1

    except:
        pass

    # counter += 1
    # if counter == 5:
    #     break

print(f'Win/Loss Success: {100*success/total_games:.3f}%')

sorted_teams = dict(sorted(teamStrength.items(), key=lambda item: item[1], reverse = True))

for i, (team_id, strength) in enumerate(list(sorted_teams.items())[:25], start=1):
    print(f"{i}. {get_team_name(team_id)}, {strength}")