from scipy import optimize
import numpy as np
from scipy import linalg

def minus_log_likelihood_function(ws, *args):
    x_data = args[0]
    y_data = args[1]
    m_dim = args[2]
    w = ws[0:m_dim-1]
    s = ws[m_dim-1]

    # pr = [
    #     -np.log(__norm_dist(x, y, w, s)) 
    #     for x, y in zip(x_data, y_data)
    # ]
    # minus_log_likelihood = np.sum(pr)

    logpr = [
        -__norm_dist_log(x, y, w, s)
        for x, y in zip(x_data, y_data)
    ]
    minus_log_likelihood = np.sum(logpr)

    return minus_log_likelihood

def calc_mle_opt(x_data, y_data, ini_ges, m_dim):
    #解析的に求まるけど無駄に最適化によって求めてみる
    ini_ges = ini_ges
    res = optimize.fmin_powell(
        minus_log_likelihood_function,
        ini_ges,
        (x_data, y_data, m_dim),
        maxiter=5000, maxfun=5000
    )
    return res[0:m_dim-1], res[m_dim-1]

def calc_mle_exact(x_data, y_data, m_dim):
    #解析解
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1]))
    y_data = y_data.reshape((-1, 1))
    A = np.dot(x_data.T, x_data)
    b = np.dot(x_data.T, y_data)
    w = linalg.solve(A, b)

    f = np.dot(x_data, w)
    s = np.sqrt(np.average(np.power(y_data - f, 2)))
    return w.flatten(), s

def plot_minus_log_likelihood_function(x_data, y_data):
    return

def __norm_dist(x, y, w, s):
    y_model = (np.matrix(w)*np.matrix(x))
    p = 1.0/(np.sqrt(2*np.pi)*s)*np.exp(-((y-y_model)*(y-y_model))/(2*s*s))
    return np.trace(p)

def __norm_dist_log(x, y, w, s):
    y_model = (np.matrix(w)*np.matrix(x))
    logp = np.log(1.0/(np.sqrt(2*np.pi)*s)) + np.trace(-((y-y_model)*(y-y_model))/(2*s*s))
    return logp



