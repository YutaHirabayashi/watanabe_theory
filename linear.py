
'''
線形回帰モデルにおける汎化誤差と経験誤差のプロット

真のモデル：P(y|x)=Norm(2x,10)

事前分布：Φ(w)=uniform, Φ(s)=uniform
尤度モデル：L(y|x,w,s)=Norm(wx,s)

'''

import pystan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_rand_from_true_distribution(x_array, num):
    '''真の分布からの乱数(x_data,y_dataの列)を得る for 汎化誤差の計算,学習データの生成'''

    #true_param
    w=2
    s=10

    x_data=[]
    y_data=[]
    for x in x_array:
        rand_y=np.random.normal(loc=w*x,scale=s,size=num)
        for y in rand_y:
            x_data.append(x)
            y_data.append(y)

    return x_data,y_data

def create_model(x_data,y_data):
    '''Dataからモデルを生成する（モデルを返す）'''
    return

def predict_distribution(model):
    '''予測モデルを用いて予測分布を計算する'''
    return

def calc_experiment_loss(model,x_data,y_data):
    '''経験誤差を計算する'''
    return

def calc_generalization_loss(model, pre_x_data, pre_y_data):
    '''汎化誤差を計算する'''
    return

def scatter_plot(x_data,y_data):
    '''データをプロットする'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data, c='red')
    plt.show()

    return

def exe():
    '''main'''
    print('start')

    x_array=np.arange(0,100,5)

    #学習データ生成
    x_data,y_data=create_rand_from_true_distribution(
        x_array=x_array,
        num=10
    )

    #学習データをplot
    scatter_plot(
        x_data=x_data,
        y_data=y_data
    )

    print('学習データ生成終了')


    


    return


if __name__ == "__main__":
    exe()




