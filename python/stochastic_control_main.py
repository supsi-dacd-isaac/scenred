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

filename = '/home/queen/Documents/ISAAC_matlab_svn/BB_forecasting/hierarchical_forecast/Data/base_forecasters_control.mat'
f = h5py.File(filename, 'r')

# parameters
forecasters = ['_hw','_relm','','rfhw']
n = 0
n_pool = 4
n_days = 2
n_forecasters = len(forecasters)

KPI_det = np.zeros((n_pool,n_pool,6,n_forecasters))
KPI_stoc = np.zeros((n_pool,n_pool,6,n_forecasters))
KPI_pre = np.zeros((n_pool,n_pool,6,n_forecasters))

KPI_det_real = np.zeros((n_pool,n_pool,6,n_forecasters))
KPI_stoc_real = np.zeros((n_pool,n_pool,6,n_forecasters))
KPI_pre_real = np.zeros((n_pool,n_pool,6,n_forecasters))

steps_size = np.array([1,1,2,3,5,7,10,15,21,31])
dt = 60*15*steps_size
f_counter = -1
for fore in forecasters:
    f_counter +=1

    # battery sizing
    p_periods = []
    for k in np.arange(6):
        p_i = f[f['bf']['y_te'][k][n]].value.T[0:96*n_days,0]
        p_periods.append(p_i)
    p_periods = np.hstack(p_periods)
    e_out_daily = np.abs(np.sum(p_periods[p_periods < 0]) * dt[0] / 3600 / n_days /6) # daily avarage produced energy in kWh

    for k in np.arange(6):

        if fore=='rfhw':
            # merge hw and rf forecasters
            y_hat_hw = f[f['bf']['y_hat_hw'][k][n]].value.T[0:96 * n_days, :] * steps_size.reshape(1, -1)
            y_hat_rf = f[f['bf']['y_hat'][k][n]].value.T[0:96 * n_days, :] * steps_size.reshape(1, -1)
            y_i_hw = f[f['bf']['y_i_hw'][k][n]].value.T[0:96 * n_days, :, :] * steps_size.reshape(1, -1, 1)
            y_i_hw = y_i_hw + 1e-6 * np.random.randn(y_i_hw.shape[0], y_i_hw.shape[1], y_i_hw.shape[2])
            y_i_rf = f[f['bf']['y_i'][k][n]].value.T[0:96 * n_days, :, :] * steps_size.reshape(1, -1, 1)
            y_i_rf = y_i_rf + 1e-6 * np.random.randn(y_i_rf.shape[0], y_i_rf.shape[1], y_i_rf.shape[2])
            y_hat = y_hat_hw
            y_hat[:,1:] = y_hat_rf[:,1:]
            y_i = y_i_hw
            y_i[:, 1:, :] = y_i_rf[:, 1:, :]
        else:
            # rename stuff
            y_hat = f[f['bf']['y_hat'+fore][k][n]].value.T[0:96*n_days,:]*steps_size.reshape(1,-1)
            y_i = f[f['bf']['y_i'+fore][k][n]].value.T[0:96*n_days,:,:]*steps_size.reshape(1,-1,1)
            y_i = y_i + 1e-6*np.random.randn(y_i.shape[0],y_i.shape[1],y_i.shape[2])

        y_te = f[f['bf']['y_te'][k][n]].value.T[0:96 * n_days, :] * steps_size.reshape(1, -1)
        data = {}
        data['y_te'] = y_te
        data['y_hat'] = y_hat
        data['scenarios'] = y_i
        data_pre = {}
        data_pre['y_te'] = y_te
        data_pre['y_hat'] = y_te
        data_pre['scenarios'] = y_i

        # set parameter pool

        c_pool = np.linspace(0.5, 2, n_pool)
        cap_pool = np.linspace(0.5, 2, n_pool)*e_out_daily
        # plot some data
        '''
        fig,ax = plt.subplots(1)
        plt.show()
        for i in np.arange(np.minimum(y_i.shape[0],100)):
            ax.clear()
            plt.plot(y_hat[i,:])
            plt.plot(y_te[i,:])
            plt.plot(np.squeeze(y_i[i,:,:]),linewidth=0.1)
            fig.canvas.draw()
            fig.canvas.flush_events()
        '''
        #-------------- BATTERY SETUP --------------------------------------

        h = y_hat.shape[1]
        pars = {'h':h,
                'ts':np.atleast_1d(dt),
                'ps':0.07,
                'pb':0.2,
                'type':'stochastic',
                'alpha':1,
                'rho':1,
                'n_final_scens':40,
                'n_init_scens':10}

        for i in np.arange(n_pool):
            for j in np.arange(n_pool):
                t1 = time()
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

                n_steps = y_te.shape[0]
                P_obs = y_te[:,[0]]
                history_stoc, cost_stoc, peak_sh_stoc, cost_real_stoc, peak_sh_real_stoc= stoc_batt.do_mpc(n_steps,P_obs,do_plots=False)
                history_pre, cost_pre, peak_sh_pre, cost_real_pre, peak_sh_real_pre= pre_batt.do_mpc(n_steps, P_obs,do_plots=False)
                history_det, cost_det, peak_sh_det, cost_real_det, peak_sh_real_det = det_batt.do_mpc(n_steps, P_obs,do_plots=False)

                fig, ax = plt.subplots(2)
                ax[0].clear()
                ax[1].clear()
                ax[0].plot(history_det['P_obs'],label='observed')
                ax[0].plot(history_stoc['P_obs_cont'],'--',label='stoc')
                ax[0].plot(history_det['P_obs_cont'],label='det')
                ax[0].plot(history_pre['P_obs_cont'],label='pre')
                ax[0].legend()

                ax[1].plot(history_stoc['SOC'], label='stoc')
                ax[1].plot(history_det['SOC_real'],label='det real')
                ax[1].plot(history_stoc['SOC_real'],label='stoc real')
                plt.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()

                # retrieve KPIs
                KPI_det[i, j,k,f_counter]  = np.sum(cost_det + peak_sh_det)
                KPI_stoc[i, j,k,f_counter] = np.sum(cost_stoc + peak_sh_stoc)
                KPI_pre[i, j, k,f_counter] = np.sum(cost_pre + peak_sh_pre)

                KPI_det_real[i, j,k,f_counter]  = np.sum(cost_real_det + peak_sh_real_det)
                KPI_stoc_real[i, j,k,f_counter] = np.sum(cost_real_stoc + peak_sh_real_stoc)
                KPI_pre_real[i, j, k,f_counter] = np.sum(cost_real_pre + peak_sh_real_pre)

                dtime = time()-t1
                print(dtime/60, c,cap,(KPI_det[i, j,k,f_counter]-KPI_stoc[i, j,k,f_counter])/KPI_stoc[i, j,k,f_counter],(KPI_det_real[i, j,k,f_counter]-KPI_stoc_real[i, j,k,f_counter])/KPI_stoc_real[i, j,k,f_counter])

                plt.close('all')

    results = {}
    results['KPT_det'] = KPI_det
    results['KPT_stoc'] = KPI_stoc
    results['KPT_pre'] = KPI_pre
    results['KPT_det_real'] = KPI_det_real
    results['KPT_stoc_real'] = KPI_stoc_real
    results['KPT_pre_real'] = KPI_pre_real
    np.save('results/results_10_40', results)

