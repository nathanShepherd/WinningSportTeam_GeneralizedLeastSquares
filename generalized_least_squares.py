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
import numpy as np

dependants = [win_perc.values, opponent_pts_per_game]

_x = np.array(df.values) # explanatory vars
_y = np.array(dependants[0]) # observed vars

def LLS(x, y, z, viz=False):

	beta = np.dot(x.T, y)
	beta = np.dot(np.linalg.inv(np.dot(x.T, x)), beta)

	if viz:
		pred = np.dot(beta.T, z)
		print( x, x.shape )
		print( y, y.shape )
		print(beta, sum(beta))
		print("Prediction: ", pred)
	
	variance = 0
	for i in range(len(x)):
		res = y[i] - np.dot(beta.T, x[i])
		res = res * res
		variance += res

	coef = variance / ((len(x) - len(x.T)))
	cov_mat = coef * np.linalg.inv(np.dot(x, x.T))

	return cov_mat, beta.T

cov = LLS(_x, _y, league_avg, True)
#print(cov)

def GLS(x, y, z):
	cov_mat, residual = LLS(x, y, None)
	omega_inv = np.linalg.inv(cov_mat)
	y_cov = np.dot(omega_inv, y)
	x_cov = np.dot(omega_inv, x)

	beta_hat = np.dot(np.linalg.inv(np.dot(x.T, x_cov)),
			  np.dot(x.T, y_cov))

	lls_y = np.dot(residual, y)

	# 	
		
	residual = np.dot(residual, z)
	pred = np.dot(beta_hat.T, z)
	adj_pred = residual + pred
	print("GLS Prediction:", pred, residual, adj_pred)
	return beta_hat

for team in _x:
	GLS(_x, _y, team)

# Covariance Mat StackExch. https://bit.ly/348bmHv


'''
# Normalize x 
for i in range(len(x)):
	for j in range(len(x[i])):
		x[i][j] = x[i][j] - league_avg[j]
		
# Normalize y
y_mu = sum(y) / len(y)
for i in range(len(y)):
	y -= y_mu
'''
'''
 # Linear Least Squares using statsmodels
print("Win %:", y)
model = sm.OLS(y, x).fit()
print(model.summary())
print("#~#~#~# L_AVG pred:",model.predict(league_avg))
# Generalized Least Squares
ols_res = model.resid # OLS residuals

print(ols_res)
res_fit = sm.OLS(list(ols_res[1:]), list(ols_res[:-1])).fit()
#rho = res_fit.params
rho = res_fit.params
print(rho)



# sigma is an n x n autocorrelation matrix for the data.
from scipy.linalg import toeplitz

order = toeplitz(np.arange(30))
#order = toeplitz(np.arange(21))
#order = toeplitz(range(len(ols_res)))

sigma = rho**order
print(sigma, sigma.shape)
'''

''' # Alternative GLS method
X = np.array(x)
print(res_fit.cov_params())
omega_inv = np.linalg.inv(sigma)

beta_pred = np.dot(omega_inv, league_avg)

#beta_pred = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(X.T, np.dot(omega_inv, X))), X.T), omega_inv), league_avg)
print(beta_pred)
'''
'''
gls_model = sm.GLS(y, x, sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary())
pred = gls_model.predict(league_avg)
print(pred, pred.shape )

coef = ['0.0415', '0.5300', '0.1123', '0.5546', '0.4904', '0.0644', '4.3499', '0.2559', '0.0673', '1.2246', '0.1978', '0.1601', '4.6855', '0.3117', '0.2729', '0.2244', '0.0075', '0.0803', '0.0017', '0.0691', '0.0006']
for i in range(len(coef)):
	coef[i] = float(coef[i]) 

print("Coefficients:", coef)


coef = np.array(coef).T
coef_pred = np.dot(coef, x[0])
print("Prediction for league average: ", coef_pred)

#print(gls_model.wendog)
'''
