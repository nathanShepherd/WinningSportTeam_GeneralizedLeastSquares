import pandas as pd

df = pd.read_csv("nba_stats_2018.csv")
df = df.drop(["G"], axis=1)

league_avg = df.iloc[30].values
df = df.drop([30], axis=0)
league_avg = league_avg[2:-1]

team_names = df[['Team']]
df = df.drop(['Team'], axis=1)

win_perc = df[["Win %"]]
df = df.drop(["Win %"], axis=1)
'''
opponent_pts_per_game = df[["PA/G"]]
df = df.drop(["PA/G"], axis=1)

team_pts = df[["PTS"]]
df = df.drop(['PTS'], axis=1)
'''
rank = df[["Rk (in terms of points)"]]
df = df.drop(["Rk (in terms of points)"], axis=1)

#league_avg = df.iloc[30].values
#df = df.drop([30], axis=0)
#league_avg = league_avg[2:-3]

#FG%, 3P%, 3PA, fta, TRB, AST, PTS, PA/G


titles = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PA/G']
for t in titles:
	print(t)
tests = ['FG%', '3P%', '3PA', 'FTA', 'TRB', 'AST', 'PTS', 'PA/G']

print(df.head)
#print(team_names)
print(league_avg, len(league_avg))

import statsmodels.api as sm
import numpy as np

def add_const(vect):
	v = [[1] for i in range(len(vect))]
	for i, row in enumerate(vect):
		for elem in row:
			v[i].append(elem)
	return v

#dependants = [win_perc.values, opponent_pts_per_game]

#_x = np.array(add_const(df.values)) # explanatory vars
_x = np.array((df.values)) # explanatory vars
_y = np.array(win_perc.values) # observed vars

def LLS(x, y, z=[None], viz=False):

	beta = np.dot(x.T, y)
	beta = np.dot(np.linalg.inv(np.dot(x.T, x)), beta)

	if viz:
		#z = add_const([z])[0]
		pred = np.dot(beta.T, z)
		print( x, x.shape )
		print( y, y.shape )
		print(beta, sum(beta), beta.shape)
		for elem in beta:
			print(str(elem)[1:-1])
		print("Prediction: ", pred)
	
	variance = 0
	for i in range(len(x)):
		res = y[i] - np.dot(beta.T, x[i])
		res = res * res
		variance += res

	coef = variance / ((len(x) - len(x.T)))
	cov_mat = coef * np.linalg.inv(np.dot(x, x.T))

	return cov_mat, beta.T

cov, lls_beta = LLS(_x, _y, league_avg, True)
#print(cov)


# Testing to see which titles have best accuracy
df_old = df
test_names = []
test_res = []
for i in range(len(titles) - 3):
	print(i)
	sample = [titles[i]]
	sample.append(tests[1])
	sample.append(tests[-2])
	sample.append(tests[-1])

	for t in titles:
		if t not in sample:
			df = df.drop([t], axis=1)

	_, beta = LLS(df.values, win_perc.values)

	summ = 0	
	for i, team in enumerate(df.values):
		summ += np.dot(beta, team) - win_perc.values[i]
		#print(sample, np.dot(beta, team) )

	print(sample, summ)
	for num in beta:
		print(str(num)[1:-1])
	
	df = df_old

def GLS(x, y, z):
	cov_mat, residual = LLS(x, y, None)
	#print(np.cov(x),cov_mat)
	cov_mat = 1/ cov_mat
	omega_inv =  np.linalg.inv(cov_mat)
	y_cov = np.dot(omega_inv, y)
	x_cov = np.dot(omega_inv, x)

	beta_hat = np.dot(np.linalg.inv(np.dot(x.T, x_cov)),
			  np.dot(x.T, y_cov))

	#lls_y = np.dot(residual, y)
	# Minimize least squares of lls prediction
		
	residual = np.dot(residual, z)
	pred = np.dot(beta_hat.T, z)
	adj_pred = residual + np.sqrt(pred)/1000
		
	#print("GLS Predict:", pred,
	#      "LLS Predict:", residual,
	#      "LLS + GLS:", adj_pred)

	#print(str(residual)[1:-1]) #LLS result
	#print("GLS" + str(pred)[1:-1]) #GLS result
	print(str(adj_pred)[1:-1])
	return beta_hat

for team in _x:
	t = GLS(_x, _y, team)

# Covariance Mat StackExch. https://bit.ly/348bmHv

df = pd.read_csv("nba_stats_2017.csv")
df = df.drop(["G"], axis=1)

team_names = df[['Team']]
df = df.drop(['Team'], axis=1)

win_perc = df[["Win%"]]
df = df.drop(["Win%"], axis=1)

rank = df[["Rk"]]
df = df.drop(["Rk"], axis=1)

_x = df.values
_y = win_perc.values

# GLS
print("GLS")
for team in _x:
	t = GLS(_x, _y, team)

# LLS
print("LLS")
for i, row in enumerate(_x):
	pred = np.dot(lls_beta, row)
	print(str(pred[0]),) 
# True value
for i, row in enumerate(_x):
	print(str(_y[i][0]))

# LLS- true value
for i, row in enumerate(_x):
	pred = np.dot(lls_beta, row)
	print(str((pred - _y[i])[0]))

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
