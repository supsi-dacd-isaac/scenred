from forecasters import RELM
import scipy.io as io
from scenred import scenred, plot_scen, plot_graph
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from _battery_controller import BatteryController
import networkx as nx
import battery
from skgarden import forest as QRF

np.random.seed(0)

#-------------- CREATE SINUSOIDAL SIGNAL --------------------------------------
n_days = 100
obs_per_day = 20
t = np.linspace(0,n_days*obs_per_day,n_days*obs_per_day)
N = len(t)
#noise = np.random.lognormal(1,1.2,N)
# make n_scen scenarios out of a detrended random walk
n_scen = 1
R = np.random.randn(N,n_scen)
W = np.zeros((N,n_scen))
for i in np.arange(n_scen):
    #stdv = np.random.randn(1)
    stdv = 0.2
    w = np.zeros((N,1))
    for j in np.arange(N-1):
        w[j+1] = w[j] + R[j,i]*stdv
    W[:,i] = w.ravel()

ma_len = int(obs_per_day*2)
Ma = np.zeros((N,N))
for i in np.arange(N):
    Ma[i,i:i+ma_len] = 1/ma_len
W_det = W - Ma.dot(W)
W_det = W_det[:N,:]

y = np.sin(t*n_days*2*np.pi/N)+ W_det[:,[0]].ravel()
x_cos = np.cos(t*n_days*2*np.pi/N)
y_past = y
# henkelize
X = np.zeros((N-2*obs_per_day,2*obs_per_day))
target = []
i=0
for e in np.arange(obs_per_day):
    win = e+np.linspace(0,N-2*obs_per_day-1,N-2*obs_per_day,dtype=int)
    X[:, [i]] = x_cos[win].reshape(-1,1)
    X[:, [i+obs_per_day]] = y_past[win].reshape(-1, 1)
    target.append(y[win+obs_per_day].reshape(-1,1))
    i+=1

dataset = {}
target = np.hstack(target)

#-------------- GENEARTE DATASET FOR ONLINE FORECASTING --------------------------------------
dataset['X_tr'] = X[0:int(N*0.8),:]
dataset['y_tr']= target[0:int(N*0.8),:]
dataset['X_te']= X[int(N*0.8):,:]
dataset['y_te'] = target[int(N*0.8):,:]

reg = 'rf'
n_regressors = 200
N_days = 5
if reg == 'RELM':
    relm = RELM(dataset['X_te'],nodes=200,n_elms=n_regressors,lamb=1)
    relm.train(dataset['X_tr'],dataset['y_tr'])
    y_hat,p,y_i = relm.predict(np.arange(obs_per_day*N_days))
else:
    rf = QRF.RandomForestRegressor(n_regressors)
    rf.fit(dataset['X_tr'],dataset['y_tr'])
    y_hat = rf.predict(dataset['X_te'][np.arange(obs_per_day*N_days),:])
    y_i = np.zeros((obs_per_day*N_days,obs_per_day,n_regressors))
    i=0
    for e in rf.estimators_:
        y_i[:,:,i]=e.predict(dataset['X_te'][np.arange(obs_per_day*N_days),:])
        i+=1


data = {}
data['y_hat'] = y_hat
data['scenarios'] = y_i+1e-6*np.random.randn(np.shape(y_i)[0],np.shape(y_i)[1],np.shape(y_i)[2])

data_pre = {}
data_pre['y_hat'] = dataset['y_te'][:obs_per_day*N_days,:]
data_pre['scenarios'] = y_i[:obs_per_day*N_days,:]

plt.figure()
plt.plot(y_hat[0,:])
plt.plot(dataset['y_te'][0,:])
plt.plot(y_i[0,:],linewidth=0.1)
#-------------- BATTERY SETUP --------------------------------------
dt = 60*15
pars = {'h':obs_per_day,
        'ts':np.atleast_1d(dt),
        'ps':0.07*1e-10,
        'pb':0.2*1e-10,
        'type':'stochastic',
        'alpha':1,
        'rho':1,
        'n_final_scens':10,
        'n_init_scens':3}

n_pool = 5
c_pool = np.linspace(1.5,2,n_pool)
cap_pool = np.linspace(1,4,n_pool)
KPI_det = np.zeros((n_pool,n_pool))
KPI_stoc = np.zeros((n_pool,n_pool))
KPI_det_real = np.zeros((n_pool,n_pool))
KPI_stoc_real = np.zeros((n_pool,n_pool))

for i in np.arange(n_pool):
    for j in np.arange(n_pool):
        c = c_pool[i]
        cap = cap_pool[j]
        # stochastic battery
        pars['type'] = 'stochastic'
        stoc_batt = battery.Battery(dataset=data, pars=pars, c_nom=c, cap_nom=cap)
        # deterministic battery
        pars['type'] = 'peak_shaving'
        det_batt = battery.Battery(dataset=data, pars=pars, c_nom=c, cap_nom=cap)
        # prescient battery
        pars['type'] = 'peak_shaving'
        pre_batt = battery.Battery(dataset=data_pre, pars=pars, c_nom=c, cap_nom=cap)

        n_steps = obs_per_day*N_days
        P_obs = dataset['y_te'][0:obs_per_day*N_days,0].reshape(-1,1)
        history_stoc, cost_stoc, peak_sh_stoc, cost_real_stoc, peak_sh_real_stoc= stoc_batt.do_mpc(n_steps,P_obs,do_plots=True)
        history_pre, cost_pre, peak_sh_pre, cost_real_pre, peak_sh_real_pre= pre_batt.do_mpc(n_steps, P_obs,do_plots=True)
        history_det, cost_det, peak_sh_det, cost_real_det, peak_sh_real_det = det_batt.do_mpc(n_steps, P_obs,do_plots=True)

        fig, ax = plt.subplots(2)
        ax[0].clear()
        ax[1].clear()
        ax[0].plot(history_det['P_obs'],label='observed')
        ax[0].plot(history_stoc['P_obs_cont'],label='stoc')
        ax[0].plot(history_det['P_obs_cont'],label='det')
        ax[0].plot(history_pre['P_obs_cont'],label='pre')

        ax[1].plot(history_stoc['SOC'], label='stoc')
        ax[1].plot(history_det['SOC_real'],label='det real')
        ax[1].plot(history_stoc['SOC_real'],label='stoc real')
        plt.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

        KPI_det[i, j]  = np.sum(cost_det + peak_sh_det)
        KPI_stoc[i, j] = np.sum(cost_stoc + peak_sh_stoc)
        KPI_det_real[i, j]  = np.sum(cost_real_det + peak_sh_real_det)
        KPI_stoc_real[i, j] = np.sum(cost_real_stoc + peak_sh_real_stoc)
        print(c,cap,(KPI_det[i, j]-KPI_stoc[i, j])/KPI_stoc[i, j],(KPI_det_real[i, j]-KPI_stoc_real[i, j])/KPI_stoc_real[i, j])
N = obs_per_day*N_days
stoc_batt.online_stochastic_plot(N)

