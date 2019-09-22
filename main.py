'''
    1次元線形回帰
        - 推定方法
            - 最尤法
            - ベイズ法
        - 汎化誤差
        - 学習誤差
        - AIC
        - WAIC
'''
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import pickle

from training_mle import calc_mle_opt
from training_mle import minus_log_likelihood_function

#真の分布からサンプルを生成する
def create_sample(n, w, s):
    x_arr = rd.uniform(0, 20, n)
    y_mean = [w*np.matrix([[x], [1]]) for x in x_arr]
    y = rd.normal(y_mean, s).flatten()
    return x_arr, y

#描画
def simple_plot(x_train, y_train, train_w=None, train_s=None, m_dim=None):
    #可視化
    plt.figure()
    plt.plot(x_train, y_train, 'o')
    plt.xlim(0,20)
    plt.ylim(0,50)

    if train_w is not None and train_w is not None:
        x_arr = np.arange(0, 20, 1)
        feature_mat = make_feature_matrix(x_arr, m_dim)
        y_arr = np.array([np.trace((np.matrix(train_w)*np.matrix(f))) for f in feature_mat])
        plt.plot(x_arr, y_arr)

    plt.savefig("test.png")
    plt.close()
    return

#特徴ベクトルの生成
def make_feature_matrix(x_data, m_dim):
    if m_dim == 3:
        feature_mat = np.array([[[x],[1]] for x in x_data])
    elif m_dim == 5:
        feature_mat = np.array([[[np.power(x,3)],[np.power(x,2)],[x],[1]] for x in x_data])
    elif m_dim == 7:
        feature_mat = np.array([[[np.power(x,5)], [np.power(x, 4)], [np.power(x,3)],[np.power(x,2)],[x],[1]] for x in x_data])
    return feature_mat

#汎化誤差(最尤法)
def calc_g_n(true_w, true_s, model_w, model_s, model_dim):
    N=10000 #モンテカルロ積分で近似する
    #真の分布による乱数を生成
    x, y = create_sample(N, true_w, true_s)

    feature_mat = make_feature_matrix(x, model_dim)
    v_sum = minus_log_likelihood_function(
        model_w.tolist() + [model_s], 
        feature_mat,y,model_dim
    )
    g_n = v_sum/N
    return g_n

#最尤解探索のための初期値を生成する
def calc_ini_ges(true_w, true_s, m_dim):
    if m_dim == 3:
        ini_ges = np.array(true_w).flatten().tolist() + [true_s] #真のモデル値を初期値として最尤解を探索
    elif m_dim == 5:
        ini_ges = [0, 0] + np.array(true_w).flatten().tolist() + [true_s]
    elif m_dim == 7:
        ini_ges = [0, 0, 0, 0] + np.array(true_w).flatten().tolist() + [true_s]
    
    return ini_ges

def mle(n, model_dim):

    #真のモデルパラメータの設定
    w = np.matrix([2.0, 5.0]) #a, b
    s = 5

    #学習用データの生成
    x_train_data, y_train_data = create_sample(n, w, s)

    #学習
    feature_mat = make_feature_matrix(x_train_data, model_dim)
    ini_ges = calc_ini_ges(w, s, model_dim)
    train_w, train_s = calc_mle_opt(feature_mat, y_train_data, ini_ges, model_dim)
    
    #図示
    simple_plot(x_train_data, y_train_data, train_w, train_s, model_dim)

    #学習誤差の計算
    feature_mat = make_feature_matrix(x_train_data, model_dim)
    t_n = 1.0/n*minus_log_likelihood_function(
        train_w.tolist() + [train_s], 
        feature_mat,y_train_data,model_dim
    )
    
    #汎化誤差の計算
    g_n = calc_g_n(
        true_w = w,
        true_s = s,
        model_w = train_w,
        model_s = train_s,
        model_dim = model_dim
    )

    return [t_n, g_n]

    #汎化誤差の推定

def mle_exp():
    sample_num = 15
    exp_num = 1
    dict_g_n = {}
    dict_t_n = {}
    for model_dim in [3, 5, 7]:
        res = [mle(sample_num, model_dim) for i in range(exp_num)]
        t_n_list = [res[i][0] for i in range(len(res))]
        g_n_list = [res[i][1] for i in range(len(res))]
        dict_g_n[str(model_dim)] = g_n_list
        dict_t_n[str(model_dim)] = t_n_list

    with open('mle_error.pickle', mode='wb') as f:
        pickle.dump(dict_g_n, f)
        pickle.dump(dict_t_n, f)

def main():
    #mle_exp()
    with open("mle_error.pickle", mode="rb") as f:
        dict_g_n = pickle.load(f)
        dict_t_n = pickle.load(f)
    

    
    
    return

if __name__ == "__main__":
    main()
    pass