

'''
線形回帰モデルにおける汎化誤差と経験誤差のプロット

真のモデル：P(y|x)=Norm(2x,10)

事前分布：Φ(w)=uniform, Φ(s)=uniform
尤度モデル：L(y|x,w,s)=Norm(wx,s)

'''

#%%
#import

import pystan
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import arviz as az
plt.style.use('default')

#%%
#定数の宣言

COMPILED_MODEL_PKL_NAME='compiled_model.pkl'
IS_COMPILE=False
W_VALUE=2
S_VALUE=10
#%%
#全関数の定義

def get_true_y_func(input_x):
    return lambda x:np.exp(-(x-input_x*W_VALUE)**2/(2*(S_VALUE**2))) / np.sqrt(2*np.pi*(S_VALUE**2))

def create_rand_from_true_distribution(x_array, num):
    '''真の分布からの乱数(x_data,y_dataの列)を得る for 汎化誤差の計算,学習データの生成'''
    x_data=[]
    y_data=[]
    for x in x_array:
        rv=scipy.stats.norm(loc=x*W_VALUE, scale=S_VALUE)
        rand_y=rv.rvs(size=num)
        for y in rand_y:
            x_data.append(x)
            y_data.append(y)

    return x_data,y_data

def compile_model_and_save():
    #stan
    model="""
    data{
        int<lower=0> N;
        int<lower=0> pred_N;
        real x_data[N];
        real y_data[N];
        real pred_x_data[pred_N];
        
    }
    parameters{
        real w;
        real<lower=0> s;
    }
    model{
        for(i in 1:N){
            y_data[i]~normal(w*x_data[i],s);
            }
    }
    generated quantities{
        real pred_y_data[pred_N];
        for(i in 1:pred_N){
            pred_y_data[i]=normal_rng(w*pred_x_data[i],s);
            }
    }
    """

    #compile
    smt=pystan.StanModel(model_code=model)

    #保存
    with open(COMPILED_MODEL_PKL_NAME, 'wb') as f:
        pickle.dump(smt, f)



def calc_estimate_and_predict(x_data, y_data, pred_x_data):
    '''Dataから学習データに対する分布推定と未知データに対する分布推定を行う'''

    #mcmc parameter
    n_itr = 5000
    n_warmup = 1000
    chains = 2

    dict_data={
        'x_data':x_data,
        'y_data':y_data,
        'pred_x_data':pred_x_data,
        'N':len(x_data),
        'pred_N':len(pred_x_data)
    }

    with open(COMPILED_MODEL_PKL_NAME, "rb") as f:
        compiled_model=pickle.load(f)
    
    fit=compiled_model.sampling(
        data=dict_data,
        iter=n_itr,
        warmup=n_warmup,
        chains=chains,
        algorithm="NUTS"
    )

    print(fit)

    return fit

def calc_kde(rand_y_list):
    '''分布推定した乱数から確率密度関数を求める'''
    kde = scipy.stats.gaussian_kde(rand_y_list)
    return kde

def calc_experiment_loss(x_data,y_data,pred_y_func_list):
    '''経験損失を計算する'''
    return

def calc_generalization_loss(x_data,true_y_func_list,pred_y_func_list):
    '''汎化損失を計算する'''
    return

def scatter_plot(x_data,y_data):
    '''データをプロットする'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data, c='red')

    return

def view_stan_result(fit):
    az.plot_density(fit, var_names=['w', 's'])
    return

def integ(x):
    '''積分テスト'''
    lambda f:(np.exp(-x**2/2)) / np.sqrt(2*np.pi)
    s,_ = scipy.integrate.quad(f,-1000,1000)
    return s

#################main#######################################

#%%
#入力xの範囲指定
x_array=np.arange(0,100,5)

#%%
#各xに対する真の確率密度関数のリストを取得
true_y_func_list=[]
for x in x_array:
    func=get_true_y_func(input_x=x)
    true_y_func_list.append(func)

#%%
#学習データの数(1点あたり)
N=3
#%%
#学習データ生成
x_data,y_data=create_rand_from_true_distribution(
    x_array=x_array,
    num=N
)
#%%
#学習データをplot
scatter_plot(
    x_data=x_data,
    y_data=y_data
)
#%%
#学習
if(IS_COMPILE):
    compile_model_and_save()

fit=calc_estimate_and_predict(
    x_data=x_data,
    y_data=y_data,
    pred_x_data=x_array
)

#%%
#学習結果の描画
view_stan_result(fit)

#%%
#それぞれの予測値の分布を得る
pred_y_func_list=[]
for x_num in range(0,len(x_array)):
    kde=calc_kde(fit['pred_y_data'][:,x_num])
    pred_y_func_list.append(kde)

#%%
