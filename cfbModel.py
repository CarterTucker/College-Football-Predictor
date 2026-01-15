import math
import cfblib as cfb


def train(yearI, yearF, gamma1, rVar, qWeek, qOffseason):
    '''
    Iterates chronologically through games and returns value of HFA, team info post-training, and total error
    
    :param yearI: when the model begins training
    :param yearF: when the model finishes training
    :param gamma1: learning rate for home field advantage
    :param gamma2: learning rate for compression
    :param rVar: predicted observation noise (high R = high trust in games)
    :param qWeek: variability of teams between weeks
    :param qOffseason: variability of teams between seasons
    '''

    # intialize internal parameters
    homeFieldAdv = 3
    compression = 30

    # read csv data and initialize team data
    statsDf = cfb.load_games('games.csv', yearI, yearF)
    teamData = cfb.initialize_teams(statsDf)

    # initialize error
    totalError = 0

    gameCount = 0
    for idx, game in statsDf.iterrows():
        
        # update team variability if the season changed
        if cfb.checkSeasonChange(statsDf, idx):
            teamData = cfb.offseasonUpdate(teamData, qOffseason)

        try:

            homeId = game['homeId']
            awayId = game['awayId']

            mHat = cfb.predictMargin(game, teamData, homeFieldAdv)

            error = cfb.compressMargin(game['margin'], compression) - mHat
            totalError += error**2

            # update Kalman parameters
            pVar = teamData[homeId]['variance'] + teamData[awayId]['variance']
            kVar = pVar/(pVar + rVar)

            # uppdate team strengths and variances
            teamData[homeId]['strength'] += kVar*error
            teamData[awayId]['strength'] -= kVar*error
            teamData[homeId]['variance'] = (1 - kVar)*teamData[homeId]['variance'] + qWeek
            teamData[awayId]['variance'] = (1 - kVar)*teamData[awayId]['variance'] + qWeek

            # update the home field advantage and compression constant with gradient descent
            if game['neutralSite'] == False:
                homeFieldAdv += gamma1*error

        except:
            pass

        # gameCount += 1
        # if gameCount % 500 == 0:  # Print every 500 games
        #     print(f"Game {gameCount}: HFA={homeFieldAdv:.2f}, "
        #           f"kVar={kVar:.4f}, sample error={error:.2f}")

    return [homeFieldAdv, compression, teamData, totalError]


def test(yearI, yearF, rVar, qWeek, qOffseason, teamInfo, homeFieldAdv, compression):
    '''
    Docstring for test
    
    :param yearI: when the model begins testing
    :param yearF: when the model finishes testing
    :param rVar: predicted observation noise (high R = high trust in games)
    :param qWeek: variability of teams between weeks
    :param qOffseason: variability of teams between seasons
    :param teamInfo: team strengths and variances from training
    :param homeFieldAdv: HFA value learned by training
    :param compression: compression rate learned by training
    '''


    # read csv data and initialize team data
    statsDf = cfb.load_games('games.csv', yearI, yearF)
    teamData = cfb.initialize_teams(statsDf, state = 'test', teamInit = teamInfo)

    # initialize error
    totalError = 0

    for idx, game in statsDf.iterrows():
        
        # update team variability if the season changed
        if cfb.checkSeasonChange(statsDf, idx):
            teamData = cfb.offseasonUpdate(teamData, qOffseason)

        try:

            homeId = game['homeId']
            awayId = game['awayId']

            mHat = cfb.predictMargin(game, teamData, homeFieldAdv)

            error = cfb.compressMargin(game['margin'], compression) - mHat
            totalError += error**2

            # update Kalman parameters
            pVar = teamData[homeId]['variance'] + teamData[awayId]['variance']
            kVar = pVar/(pVar + rVar)

            # uppdate team strengths and variances
            teamData[homeId]['strength'] += kVar*error
            teamData[awayId]['strength'] -= kVar*error
            teamData[homeId]['variance'] = (1 - kVar)*teamData[homeId]['variance'] + qWeek
            teamData[awayId]['variance'] = (1 - kVar)*teamData[awayId]['variance'] + qWeek

        except:
            pass

    return [teamData, totalError]