import pandas as pd
import os
from datetime import datetime, timezone
import math

# define date format and learning rate
fmt = '%Y-%m-%dT%H:%M:%S.%fZ'
beta = 0.095
HFA = 4.9059

# obtain a dataframe from the csv cfb data file
df = pd.read_csv('games.csv')

# obtain a dictionary matching team IDs to team names
teamInfo = {}
for _, game in df.iterrows():

    teamId = game['homeId']
    teamName = game['homeTeam']

    if teamId not in teamInfo.keys():
        teamInfo[teamId] = teamName

# initiate team ratings
teamStrength = {}
for teamId in teamInfo.keys():

    teamStrength[teamId] = 0

# run the model and collect model data
residuals = []
totalGames = 0
success = 0

for index, game in df.iterrows():

    # check if there is a new season and update team strengths
    previousGameDate = df.iloc[index - 1]['startDate']
    currentGameDate = game['startDate']
    dDate = (datetime.strptime(currentGameDate, fmt).replace(tzinfo=timezone.utc) - datetime.strptime(previousGameDate, fmt).replace(tzinfo=timezone.utc)).days

    if dDate>150:
        for teamId in teamStrength.keys():
            teamStrength[teamId] *= 0.8

    try:
        homeId = game['homeId']
        awayId = game['awayId']

        expectedMargin = teamStrength[homeId] - teamStrength[awayId]
        if game['neutralSite'] == False:
            expectedMargin += HFA

        error = game['margin'] - expectedMargin

        teamStrength[homeId] += beta*error
        teamStrength[awayId] -= beta*error

        residuals.append(error)
        totalGames += 1

        if (expectedMargin > 0 and game['margin'] > 0) or (expectedMargin < 0 and game['margin'] < 0) or (expectedMargin == 0 and game['margin'] == 0):
            success += 1

    except Exception as e:
        pass
        # print(f'Error:\n{e}')


meanRSq = sum(residuals)/len(residuals)
print(meanRSq) 

# rSq = [r**2 for r in residuals]
# sigmaSq = 1/(totalGames - len(teamInfo.keys()))*sum(rSq)
# sigma = math.sqrt(sigmaSq)

# print(sigma, 1/sigmaSq)

print(f'Win/Loss Success: {100*success/totalGames:.3f}%')

sorted_teams = dict(sorted(teamStrength.items(), key=lambda item: item[1], reverse = True))

for i, (team_id, strength) in enumerate(list(sorted_teams.items())[:25], start=1):
    print(f"{i}. {teamInfo[team_id]}, {strength}")
