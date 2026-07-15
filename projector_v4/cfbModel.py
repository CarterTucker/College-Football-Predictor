import cfbModelFramework as cfb
import traceback

def train(year_i, year_f, gamma, R_var, Q_week, Q_offseason):
    '''
    Docstring for train
    
    :param year_i: year model training begins
    :param year_f: year model training ends
    :param gamma: describes the learning rate for the Home Field Advantage
    :param R_var: predicted observation noise (higher R means you trust game results more)
    :param Q_week: predicted season noise (higher Q_week means that teams aren't punished for noisy performance)
    :param Q_offseason: predicted offseason noise (higher Q_offseason reduces season-to-season retention)

    :returns: value of HFA, team strengths at the end of training, sum of squared errors
    '''
    
    # take a guess at the value of home field advantage
    homeFieldAdvantage = 3

    # read the csv file and initialize team data 
    statsDf = cfb.loadGames('games.csv', year_i, year_f)
    teamInfo = cfb.initializeTeams(statsDf, state = 'train')

    # keep track of error
    totalError = 0

    # loop through the games to train the model
    for index, game in statsDf.iterrows():

        # if there has been a new season, update team strengths
        if cfb.checkSeasonChange(statsDf, index):
            teamInfo = cfb.seasonUpdate(teamInfo, Q_offseason)

        try:

            # identify the teams that played the game
            homeId = game['homeId']
            awayId = game['awayId']

            # calculate an expected margin
            expectedMargin = teamInfo[homeId]['strength'] - teamInfo[awayId]['strength']
            if game['neutralSite'] == False:
                expectedMargin += homeFieldAdvantage

            error = cfb.compressMargin(game['margin']) - expectedMargin
            totalError += error**2

            # update parameters (update beta by Ralman, HFA by gradient descent)
            if game['neutralSite'] == False:
                homeFieldAdvantage += gamma*error*1 # HFA is given value 1 and weight homeFieldAdvantage
            P_var = teamInfo[homeId]['variance'] + teamInfo[awayId]['variance']
            K_var = P_var/(P_var + R_var)

             # update strengths and variances
            teamInfo[homeId]['strength'] += K_var*error
            teamInfo[awayId]['strength'] -= K_var*error

            teamInfo[homeId]['variance'] = (1 - K_var)*teamInfo[homeId]['variance'] + Q_week
            teamInfo[awayId]['variance'] = (1 - K_var)*teamInfo[awayId]['variance'] + Q_week


            # # keep track of the success of the model
            # numGames += 1

            # if (expectedMargin > 0 and game['margin'] > 0) or (expectedMargin < 0 and game['margin'] < 0) or (expectedMargin == 0 and game['margin'] == 0):
            #     numSuccesses += 1

        except Exception as e:
            # traceback.print_exc()
            # print(f'Error:\n{e}')
            pass

    return [homeFieldAdvantage, teamInfo, totalError]

def test(year_i, year_f, R_var, Q_week, Q_offseason, teamInfo, homeFieldAdvantage):
    '''
    Docstring for test
    
    :param year_i: year model training begins
    :param year_f: year model training ends
    :param gamma: describes the learning rate for the Home Field Advantage
    :param R_var: predicted observation noise (higher R means you trust game results more)
    :param Q_week: predicted season noise (higher Q_week means that teams aren't punished for noisy performance)
    :param Q_offseason: predicted offseason noise (higher Q_offseason reduces season-to-season retention)
    :param teamInfo: a dictionary which is a package of information about team strengths, variances, and names
    :param homeFieldAdvantage: the value of HFA learned by the model

    :returns: some of squared differences in margins 
    '''

    # read the csv file and initialize team data 
    statsDf = cfb.loadGames('games.csv', year_i, year_f)

    # initialize the model based on previous training
    teamInfo = cfb.initializeTeams(statsDf, state = 'test', teamInit = teamInfo)

    # keep track of error
    totalError = 0

    # loop through the games to train the model
    for index, game in statsDf.iterrows():

        # if there has been a new season, update team strengths
        if cfb.checkSeasonChange(statsDf, index):
            teamInfo = cfb.seasonUpdate(teamInfo, Q_offseason)

        try:

            # identify the teams that played the game
            homeId = game['homeId']
            awayId = game['awayId']

            # calculate an expected margin
            expectedMargin = teamInfo[homeId]['strength'] - teamInfo[awayId]['strength']
            if game['neutralSite'] == False:
                expectedMargin += homeFieldAdvantage

            error = cfb.compressMargin(game['margin']) - expectedMargin
            totalError += abs(error)

            # update parameters (update beta (K) by Ralman)
            P_var = teamInfo[homeId]['variance'] + teamInfo[awayId]['variance']
            K_var = P_var/(P_var + R_var)

            # update strengths and variances
            teamInfo[homeId]['strength'] += K_var*error
            teamInfo[awayId]['strength'] -= K_var*error

            teamInfo[homeId]['variance'] = (1 - K_var)*teamInfo[homeId]['variance'] + Q_week
            teamInfo[awayId]['variance'] = (1 - K_var)*teamInfo[awayId]['variance'] + Q_week


        except Exception as e:
            # traceback.print_exc()
            # print(f'Error:\n{e}')
            pass

    return [teamInfo, totalError]