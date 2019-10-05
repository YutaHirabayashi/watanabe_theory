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
import pandas as pd
import seaborn as sns

from training_mle import calc_mle_opt
from training_mle import minus_log_likelihood_function
from training_mle import calc_mle_exact

#分布からサンプルを生成する
def create_sample(n, w, s):
    model_dim = w.shape[1] + 1
    x_arr = rd.uniform(0, 20, n)
    feature_arr = make_feature_matrix(x_arr, model_dim)
    y_mean = [w*np.matrix(x) for x in feature_arr]
    y = rd.normal(y_mean, s).flatten()
    return x_arr, y

#描画
def simple_plot(x_train, y_train, train_w=None, train_s=None, m_dim=None):
    #可視化
    plt.figure()
    plt.plot(x_train, y_train, 'o')
    plt.xlim(0,20)

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

#AIC
def calc_aic(t_n, n, m_dim):
    aic = t_n + m_dim / n
    return aic

#最尤解探索のための初期値を生成する
def calc_ini_ges(true_w, true_s, m_dim):
    if m_dim == 3:
        ini_ges = np.array(true_w).flatten().tolist() + [true_s] #真のモデル値を初期値として最尤解を探索
    elif m_dim == 5:
        ini_ges = [0, 0] + np.array(true_w).flatten().tolist() + [true_s]
    elif m_dim == 7:
        ini_ges = [0, 0, 0, 0] + np.array(true_w).flatten().tolist() + [true_s]
    
    return ini_ges

#最尤法に置ける実験内容
def mle(model_dim, x_train_data, y_train_data, true_w, true_s):

    #学習
    feature_mat = make_feature_matrix(x_train_data, model_dim)
    ini_ges = calc_ini_ges(true_w, true_s, model_dim)
    #train_w, train_s = calc_mle_opt(feature_mat, y_train_data, ini_ges, model_dim)
    
    train_w, train_s = calc_mle_exact(feature_mat, y_train_data, model_dim)
    #図示
    simple_plot(x_train_data, y_train_data, train_w, train_s, model_dim)

    #学習誤差の計算
    n = x_train_data.shape[0]
    feature_mat = make_feature_matrix(x_train_data, model_dim)
    t_n = 1.0/n*minus_log_likelihood_function(
        train_w.tolist() + [train_s], 
        feature_mat,y_train_data,model_dim
    )
    
    #汎化誤差の計算
    g_n = calc_g_n(
        true_w = true_w,
        true_s = true_s,
        model_w = train_w,
        model_s = train_s,
        model_dim = model_dim
    )

    #汎化誤差の推定
    aic = calc_aic(t_n, n, model_dim)
    return t_n, g_n, aic

#最尤法の実験を行う
def mle_exp():
    exp_num = 1000
    pd_index = ['sample_num', 'model_dim', 'exp_num', 't_n', 'g_n', 'aic']
    pd_list = []

    #真のモデルパラメータの設定
    w = np.matrix([2.0, 5.0]) #a, b
    s = 5

    for sample_num in [80, 120, 240]:
        for i in range(exp_num):
            #学習用データの生成
            x_train_data, y_train_data = create_sample(sample_num, w, s)
            for model_dim in [5,3]:
                t_n, g_n, aic = mle(model_dim, x_train_data, y_train_data, w, s)
                ser = pd.Series(
                    [sample_num, model_dim, i, t_n, g_n, aic],
                    index = pd_index
                )
                pd_list.append(ser)
    df = pd.concat(pd_list, axis=1).T
    df.to_pickle("mle_res.pkl")


def main():
    mle_exp()
    df = pd.read_pickle("mle_res.pkl")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    #汎化誤差とAICの差
    df["g_n-aic"] = df["g_n"] - df["aic"]

    for y in ["t_n", "g_n", "g_n-aic"]:
        sns.boxenplot(
            x='sample_num', 
            y=y, 
            data=df[(df["sample_num"]!=5) & (df["sample_num"]!=20)], 
            hue="model_dim"
        )

        plt.savefig(y + ".png")
        plt.close()

    #モデル１とモデル２の差分を調べる
    dif_df = pd.merge(
        df[df["model_dim"]==3], 
        df[df["model_dim"]==5],
        on=["sample_num", "exp_num"])

    dif_df["dif_aic"] = dif_df["aic_x"]-dif_df["aic_y"]
    dif_df["dif_g_n"] = dif_df["g_n_x"]-dif_df["g_n_y"]

    for y in ["dif_aic", "dif_g_n"]:
        sns.boxenplot(
            x='sample_num', 
            y=y, 
            data=dif_df[(dif_df["sample_num"]!=5) & (dif_df["sample_num"]!=20)]
        )
        plt.savefig(y + ".png")
        plt.close()

        for n in [80, 120, 240]:
            sns.distplot(dif_df[dif_df["sample_num"]==n][y], bins=15, kde=False)
            plt.savefig("dist_"+str(n)+"_"+y+".png")
            plt.close()

if __name__ == "__main__":
    main()
    pass