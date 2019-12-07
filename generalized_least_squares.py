import pandas as pd

df = pd.read_csv("nba_stats_2018.csv")
df = df.drop(["G"], axis=1)

league_avg = df.iloc[30].values
df = df.drop([30], axis=0)
league_avg = league_avg[2:-3]

team_names = df[['Team']]
df = df.drop(['Team'], axis=1)

win_perc = df[["Win %"]]
df = df.drop(["Win %"], axis=1)

opponent_pts_per_game = df[["PA/G"]]
df = df.drop(["PA/G"], axis=1)

team_pts = df[["PTS"]]
df = df.drop(['PTS'], axis=1)

rank = df[["Rk (in terms of points)"]]
df = df.drop(["Rk (in terms of points)"], axis=1)




print(df.head)
#print(team_names)
print(league_avg, len(league_avg))

import statsmodels.api as sm

x = df.values # explanatory vars
y = win_perc.values # observed vars
print("Win %:", y)
model = sm.OLS(y, x).fit()
print("#~#~#~# L_AVG pred:",model.predict(league_avg))
# Generalized Least Squares
ols_res = model.resid # OLS residuals
print(ols_res)
res_fit = sm.OLS(list(ols_res[1:]), list(ols_res[:-1])).fit()
rho = res_fit.params
print(rho)

# sigma is an n x n autocorrelation matrix for the data.
from scipy.linalg import toeplitz
import numpy as np

order = toeplitz(np.arange(30))
sigma = rho**order
print(sigma, sigma.shape)

gls_model = sm.GLS(y, x, sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary())
pred = gls_model.predict(league_avg)
print(pred, pred.shape )
