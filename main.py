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

from training import calc_mle_opt

def create_sample(n, w, s):
    x_arr = rd.uniform(0, 20, n)
    y_mean = [w*np.matrix([[x], [1]]) for x in x_arr]
    y = rd.normal(y_mean, s).flatten()
    return x_arr, y

def simple_plot(x_train, y_train):
    #可視化
    plt.figure()
    plt.plot(x_train, y_train, 'o')
    plt.xlim(0,20)
    plt.ylim(0,50)
    plt.savefig("test.png")
    plt.close()
    return

def main():

    #真のモデルパラメータの設定
    w = np.matrix([2.0, 5.0]) #a, b
    s = 5

    #学習用データの生成
    x_train, y_train = create_sample(20, w, s)
    simple_plot(x_train, y_train)

    #学習
    res = calc_mle_opt(x_train, y_train)

    #学習誤差の計算

    #汎化誤差の計算

    #汎化誤差の推定

    return

if __name__ == "__main__":
    main()
    pass