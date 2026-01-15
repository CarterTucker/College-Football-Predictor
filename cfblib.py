import pandas
import math

def load_games(filename, yearI, yearF):
    '''
    Returns a dataframe of College Football data between defined years from a CSV file.
    
    :param filename: CSV filename
    :param yearI: initial year (inclusive)
    :param yearF: final year (inclusive)
    '''

    df = pandas.read_csv(filename)
    dfFiltered = df[df['season'].between(yearI, yearF, inclusive='both')].reset_index(drop = True)
    
    return dfFiltered


def initialize_teams(df, state = 'train', teamInit = None):
    '''
    Returns a dictionary mapping team names, strengths, and variances to team IDs
    
    :param df: CFB dataframe
    :param state: takes values train and test and controls
    :param teamData: if in the test state, initialize team data with teamData, collected from the end of training
    '''

    teamInitDict = {}

    for _, game in df.iterrows():

        teamId = game['homeId']
        teamName = game['homeTeam']

        if teamId not in teamInitDict.keys():

            teamInitDict[teamId] = {}
            teamInitDict[teamId]['name'] = teamName

            if state == 'train':

                teamInitDict[teamId]['strength'] = 0
                teamInitDict[teamId]['variance'] = 50
            
            elif state == 'test':

                if teamId in teamInit.keys():

                    teamInitDict[teamId]['strength'] = teamInit[teamId]['strength']
                    teamInitDict[teamId]['variance'] = teamInit[teamId]['variance']
                
                else:

                    teamInitDict[teamId]['strength'] = 0
                    teamInitDict[teamId]['variance'] = 50
            
            else:
                print('Invalid initialization state')
        
    return teamInitDict


def checkSeasonChange(df, idx):
    '''
    Returns true if the index is the first game of a new season; returns false otherwise
    
    :param df: CFB dataframe
    :param idx: index of the current game in the model
    '''

    if idx == 0:
        return False
    
    if df.iloc[idx - 1]['season'] != df.iloc[idx]['season']:
        return True
    else:
        return False
    

def offseasonUpdate(teamData, qOffseason):
    '''
    Updates team data to account for changes between seasons
    
    :param teamData: dictionary of team data
    :param qOffseason: quantified variability from the offseason
    '''

    for teamId in teamData.keys():
        teamData[teamId]['variance'] += qOffseason

    return teamData


def predictMargin(game, teamData, homeFieldAdv):
    '''
    Returns the expected margin of a game relative to the home team
    
    :param game: data package from the dataframe describing the game
    :param homeFieldAdv: value of the home field advantage
    '''

    expMargin = teamData[game['homeId']]['strength'] - teamData[game['awayId']]['strength']

    if not game['neutralSite']:
        expMargin += homeFieldAdv

    return expMargin


def compressMargin(margin, k):
    '''
    Uses a sigmoid transformation to compress the true margin of a game
    
    :param margin: true margin of the game
    :param k: compression factor (higher k mean higher compression)
    '''

    return k*math.tanh(margin/k)