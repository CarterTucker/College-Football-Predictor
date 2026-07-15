import cfb_predictor_lib as cfb

# read the csv file
df = cfb.loadGames('games.csv')

# define a dictinary which will carry the name, strength, and variance of a team
teamInfo = cfb.initializeTeams(df)

# define some parameters which will learn alongside the model
homeFieldAdvantage = 3

# define some parameters to define the behavior of the model
gamma = 0.0005
R_var = 225
Q_week = 0.5
Q_offseason = 50

# keep track of some stats to quantify the success of the model
numSuccesses = 0
numGames = 0


# loop through the games to train the model
for index, game in df.iterrows():

    # if there has been a new season, update team strengths
    if cfb.checkSeasonChange(df, index):
        teamInfo = cfb.seasonUpdate(teamInfo, Q_offseason)

    try:

        # identify the teams that played the game
        homeId = game['homeId']
        awayId = game['awayId']

        # calculate an expected margin
        expectedMargin = teamInfo[homeId]['strength'] - teamInfo[awayId]['strength']
        if game['neutralSite'] == False:
            expectedMargin += homeFieldAdvantage

        error = game['margin'] - expectedMargin


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


        # keep track of the success of the model
        if game['season'] > 2010:
            numGames += 1

            if (expectedMargin > 0 and game['margin'] > 0) or (expectedMargin < 0 and game['margin'] < 0) or (expectedMargin == 0 and game['margin'] == 0):
                numSuccesses += 1

    except Exception as e:
        pass
        # print(f'Error:\n{e}')


print(f'Win/Loss Success: {100*numSuccesses/numGames:.3f}%')

# Sort by strength (highest first)
sorted_teams = dict(sorted(teamInfo.items(), key=lambda x: x[1]['strength'], reverse=True))

# Print top 25
for i, (team_id, info) in enumerate(list(sorted_teams.items())[:25], start=1):
    print(f"{i}. {info['name']}, {info['strength']:.2f}")

print(f'HFA: {homeFieldAdvantage}')
print(f'Weekly Var.: {Q_week}')
print(f'Offseason Var.: {Q_offseason}')
print(f'R: {R_var}')
print("Team strengths")

'''
-> Run the model from 1990 to 2015.
-> Keep track of 
   o HFA
   o Q_week
   o Q_offseason
   o R
   o Team strengths
-> Run the model from 2015 to 2025
   o Use the previous parameters
   o Initiate the model with 2015 strengths
   o Sum squared errors
-> Adjust values of Q and R freely and repeat the process
in order to minimize the sum of squared errors in the testing phase

*** Need to come up with a way to run training on certain years, keep parameters,
test on certain years, and evaluate error easily. I'll probably repeat this many 
times in changing R and Q

*** One file with core functions, what cfb_predictor_lib does
    One file with the testing and training processes
    One file which runs the training and then the testing in gradient descent to
    optimize each hyperparameter.
'''