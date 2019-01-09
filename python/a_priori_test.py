from forecasters import RELM
import scipy.io as io
from scenred import scenred, plot_scen, plot_graph
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from _battery_controller import BatteryController
import networkx as nx
import battery
from skgarden import forest as QRF
import h5py
from time import time

np.random.seed(0)

filename = '/home/queen/Documents/ISAAC_matlab_svn/BB_forecasting/hierarchical_forecast/Data/optisim_dataset.mat'
dataset = io.loadmat(filename)
PV = dataset['dataset'][0,0]['PV']
UN = dataset['dataset'][0,0]['UN']
HP = dataset['dataset'][0,0]['HP']
# create profiles (take 6 couples of days)
Pm = PV+UN+HP
n_folds = 6
profile_n = 0
N_steps = np.linspace(10,96,9,dtype=int)
n_pool = 4
KPI_c = np.zeros((n_pool,n_pool,len(N_steps),n_folds))
KPI_s = np.zeros((n_pool,n_pool,len(N_steps),n_folds))
KPI = np.zeros((n_pool,n_pool,len(N_steps),n_folds))
times = np.zeros((n_pool,n_pool,len(N_steps),n_folds))

for s in np.arange(N_steps.shape[0]):
    # chose log-spaced steps
    ss = 0
    eps = 0
    while ss < 96:
        eps += 1e-2
        steps = np.asanyarray(np.ceil(np.diff(np.exp(np.linspace(0, np.log(10 + eps), N_steps[s]+1)))),int)
        ss = steps.sum()
        ss = int(ss)

    # define step sizes
    if s==N_steps.shape[0]:
        dt = 60 * 15
        h = 96
    else:
        dt = steps*60*15
        h = N_steps[s]
    pars = {'h': h,
            'ts': np.atleast_1d(dt),
            'ps': 0.07,
            'pb': 0.2,
            'type': 'stochastic',
            'alpha': 1,
            'rho': 1,
            'n_final_scens': 20,
            'n_init_scens': 5}
    # build mean operator
    M = np.zeros((len(dt), 96))
    k = 0
    for i in np.arange(len(dt)):
        M[i, k:k + int(steps[i])] = 1/int(steps[i])
        k += int(steps[i])

    # define folds and run prescient control for each fold
    P_folds = []
    P_folds_0 = []
    for f in np.arange(n_folds):
        start = int(Pm.shape[0] / n_folds) * f
        P_fold = np.zeros((2 * 96, 96))
        for k in np.arange(P_fold.shape[0]):
            window = start + k + np.arange(96)
            P_fold[k, :] = Pm[window, profile_n]
        P_folds_0.append(P_fold[:, 0])
        P_folds.append(P_fold)

    p_periods = np.hstack(P_folds_0)
    e_out_daily = np.abs(
        np.sum(p_periods[p_periods < 0]) * dt[0] / 3600 / 2 / n_folds)  # daily avarage produced energy in kWh

    c_pool = np.linspace(0.5, 2, n_pool)
    cap_pool = np.linspace(0.5, 2, n_pool) * e_out_daily

    for f in np.arange(n_folds):
        P_hat = M.dot(P_folds[f].T).T
        data_pre = {}
        data_pre['y_te'] = P_hat
        data_pre['y_hat'] = P_hat
        data_pre['scenarios'] = np.reshape(P_hat,(P_hat.shape[0],P_hat.shape[1],1))

        # prescient battery
        pars['type'] = 'peak_shaving'

        n_steps = P_hat.shape[0]
        for i in np.arange(n_pool):
            for j in np.arange(n_pool):
                c = c_pool[i]
                cap = cap_pool[j]
                t1 = time()
                pre_batt = battery.Battery(dataset=data_pre, pars=pars, c_nom=c, cap_nom=cap)
                history_pre, cost_pre, peak_sh_pre, cost_real_pre, peak_sh_real_pre = pre_batt.do_mpc(n_steps, P_hat[:,[0]],
                                                                                             do_plots=False)
                '''  
                fig, ax = plt.subplots(1)
                ax.clear()
                ax.plot(history_pre['P_obs'],label='observed')
                ax.plot(history_pre['P_obs_cont'],'--',label='controlled')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()
'''
                KPI_c[i, j, s, f] = np.sum(cost_pre)/4 # transform in $/kWh
                KPI_s[i, j, s, f] = np.sum(peak_sh_pre)
                KPI[i, j, s, f] = np.sum(cost_pre) + np.sum(peak_sh_pre)
                dtime = time() - t1
                print(dtime/60,KPI[i,j,s,f],c,cap,N_steps[s])
                times[i, j, s, f] = dtime
        plt.close('all')

results = {}
results['KPI'] = KPI
results['KPI_c'] = KPI_c
results['KPI_s'] = KPI_s
results['times'] = times

np.save('results/a_priori_results_2', results)