import cfbModel
from scipy.optimize import minimize


trainYears = [2010, 2024]
testYears = [trainYears[1]+1, 2025]
bounds = [(1e-6, 1e-4),
          (50, 500),
          (0.01, 3),
          (10, 100)]


def evaluateParamError(params):

    gamma1, R_var, Q_week, Q_offseason = params

    homeFieldAdvantage, compression, trainTeamInfo, trainTotalError = cfbModel.train(trainYears[0], trainYears[1], gamma1, R_var, Q_week, Q_offseason)
    testTeamInfo, totalTestError = cfbModel.test(testYears[0], testYears[1], R_var, Q_week, Q_offseason, trainTeamInfo, homeFieldAdvantage, compression)

    return totalTestError

def showRankings(params):

    gamma1, R_var, Q_week, Q_offseason = params

    homeFieldAdvantage, compression, trainTeamInfo, trainTotalError = cfbModel.train(trainYears[0], trainYears[1], gamma1, R_var, Q_week, Q_offseason)
    testTeamInfo, totalTestError = cfbModel.test(testYears[0], testYears[1], R_var, Q_week, Q_offseason, trainTeamInfo, homeFieldAdvantage, compression)

    # Sort by strength (highest first)
    sorted_teams = dict(sorted(testTeamInfo.items(), key=lambda x: x[1]['strength'], reverse=True))

    # Print top 25
    for i, (team_id, info) in enumerate(list(sorted_teams.items())[:25], start=1):
        print(f"{i}. {info['name']}, {info['strength']:.2f}")

    print(f'HFA: {homeFieldAdvantage}')


x0 = [1e-5, 150, 1, 50]
res = minimize(evaluateParamError, x0, method='L-BFGS-B', bounds = bounds)

print("Optimization complete!")
print(f'Parameters: {res.x}')
showRankings(res.x)

