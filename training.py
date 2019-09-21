from scipy import optimize
import numpy as np

def norm_dist(x, y, w, s):
    y_model = (np.matrix(w)*np.matrix([[x],[1]]))
    p = 1.0/(np.sqrt(2*np.pi)*s)*np.exp(-((y-y_model)*(y-y_model))/(2*s*s))
    return np.trace(p)

def minus_log_likelihood_function(ws, *args):
    x_data = args[0]
    y_data = args[1]
    w = ws[0:2]
    s = ws[2]

    pr = [
        -np.log(norm_dist(x, y, w, s)) 
        for x, y in zip(x_data, y_data)
    ]

    minus_log_likelihood = np.sum(pr)
    return minus_log_likelihood


def calc_mle_opt(x_data, y_data):
    #解析的に求まるけど無駄に最適化によって求めてみる
    ini_ges = [10.0, 10.0, 5.0]
    minus_log_likelihood_function(ini_ges, x_data, y_data)
    res = optimize.fmin(
        minus_log_likelihood_function,
        ini_ges,
        (x_data, y_data)
    )
    return res.x

def plot_minus_log_likelihood_function():
    return



