import cfbModel
from scipy.optimize import minimize


trainYears = [2010, 2020]
testYears = [trainYears[1], 2025]


def evaluateParamError(params):

    gamma, R_var, Q_week, Q_offseason = params

    homeFieldAdvantage, trainTeamInfo, trainTotalError = cfbModel.train(trainYears[0], trainYears[1], gamma, R_var, Q_week, Q_offseason)
    testTeamInfo, totalTestError = cfbModel.test(testYears[0], testYears[1], R_var, Q_week, Q_offseason, trainTeamInfo, homeFieldAdvantage)

    return totalTestError

def showRankings(params):

    gamma, R_var, Q_week, Q_offseason = params

    homeFieldAdvantage, trainTeamInfo, trainTotalError = cfbModel.train(trainYears[0], trainYears[1], gamma, R_var, Q_week, Q_offseason)
    testTeamInfo, totalTestError = cfbModel.test(testYears[0], testYears[1], R_var, Q_week, Q_offseason, trainTeamInfo, homeFieldAdvantage)

    # Sort by strength (highest first)
    sorted_teams = dict(sorted(testTeamInfo.items(), key=lambda x: x[1]['strength'], reverse=True))

    # Print top 25
    for i, (team_id, info) in enumerate(list(sorted_teams.items())[:25], start=1):
        print(f"{i}. {info['name']}, {info['strength']:.2f}")


x0 = [0.002, 225, 0.5, 50]
res = minimize(evaluateParamError, x0, method='Nelder-Mead', options={
        'xatol': 1e-2,          # still reasonable for parameters
        'fatol': 5000,          # matches scale of your error
        'maxiter': 200           # optional
    })

print("Optimization complete!")
print(f'Parameters: {res.x}')
showRankings(res.x)