import pandas as pd
from datetime import datetime, timezone

# define date format and learning rate
fmt = '%Y-%m-%dT%H:%M:%S.%fZ'

# load the games
def loadGames(filename):
    return pd.read_csv(filename)

# build the team dictionary
def initializeTeams(df):
    teamInfo = {}
    
    for _, game in df.iterrows():
        
        teamId = game['homeId']
        teamName = game['homeTeam']

        if teamId not in teamInfo.keys():
            teamInfo[teamId] = {}
            teamInfo[teamId]['name'] = teamName
            teamInfo[teamId]['strength'] = 0
            teamInfo[teamId]['variance'] = 100
    
    return teamInfo

# check if new season
def checkSeasonChange(df, index):

    if index == 0:
        return False

    previousGameDate = df.iloc[index - 1]['startDate']
    currentGameDate = df.iloc[index]['startDate']
    dDate = (datetime.strptime(currentGameDate, fmt).replace(tzinfo=timezone.utc) - datetime.strptime(previousGameDate, fmt).replace(tzinfo=timezone.utc)).days

    if dDate>150:
        return True
    else:
        return False

# apply the new-season changes
def seasonUpdate(teamInfo, Q_offseason):

    # eventually add more terms about returning talent, etc.
    for teamId in teamInfo.keys():
        # teamInfo[teamId]['strength'] *= 0.8
        teamInfo[teamId]['variance'] += Q_offseason

    return teamInfo

