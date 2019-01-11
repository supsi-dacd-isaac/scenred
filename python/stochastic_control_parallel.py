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
import multiprocessing as mp


np.random.seed(0)


def fileReader(qin, qout):
    print('I read the file here')
    filename = '/home/queen/Documents/ISAAC_matlab_svn/BB_forecasting/hierarchical_forecast/Data/base_forecasters_control.mat'
    f = h5py.File(filename, 'r')

    while True:
        n = qin.get()
        data = {}
        keys = list(f['bf'].keys())
        keys = [keys[i] for i in [0,1,2,6,7,8,9,10,11,12]]
        for i in np.arange(len(keys)):
            temp = []
            key_i = keys[i]
            for k in np.arange(6):
                temp.append(f[f['bf'][key_i][k][n]].value)

            data[key_i] = temp
        print('got message from ' + str(n))
        qout[n].put(data)

def init(qfr, qw):
    do_something.qin = qw
    do_something.qout = qfr

def do_something(i):
    qin = do_something.qin[i]
    qout = do_something.qout
    qout.put(i)
    data_input = qin.get()
    print('hello I am job %d and I have received some data' % (i))

    forecasters = ['rfhw', '_hw', '_relm']
    n_days = 2
    n_final_scens = np.linspace(20,120,11,dtype=int)
    #n_final_scens = np.linspace(1, 2, 2, dtype=int)

    n_init_scen = 10
    len_scen = len(n_final_scens)
    n_forecasters = len(forecasters)

    KPI_det = np.zeros((6,n_forecasters,len_scen))
    KPI_stoc = np.zeros((6,n_forecasters,len_scen))
    KPI_pre = np.zeros((6,n_forecasters,len_scen))
    
    KPI_det_real = np.zeros((6,n_forecasters,len_scen))
    KPI_stoc_real = np.zeros((6,n_forecasters,len_scen))
    KPI_pre_real = np.zeros((6,n_forecasters,len_scen))
    
    KPI_c_det = np.zeros((6,n_forecasters,len_scen))
    KPI_c_stoc = np.zeros((6,n_forecasters,len_scen))
    KPI_c_pre = np.zeros((6,n_forecasters,len_scen))
    
    KPI_c_det_real = np.zeros((6,n_forecasters,len_scen))
    KPI_c_stoc_real = np.zeros((6,n_forecasters,len_scen))
    KPI_c_pre_real = np.zeros((6,n_forecasters,len_scen))
    
    KPI_p_det = np.zeros((6,n_forecasters,len_scen))
    KPI_p_stoc = np.zeros((6,n_forecasters,len_scen))
    KPI_p_pre = np.zeros((6,n_forecasters,len_scen))
    
    KPI_p_det_real = np.zeros((6,n_forecasters,len_scen))
    KPI_p_stoc_real = np.zeros((6,n_forecasters,len_scen))
    KPI_p_pre_real = np.zeros((6,n_forecasters,len_scen))
    
    times_stoc = np.zeros((6,n_forecasters,len_scen))
    times_pre = np.zeros((6,n_forecasters,len_scen))
    
    steps_size = np.array([1,1,2,3,5,7,10,15,21,31])
    dt = 60*15*steps_size

    n_counter = -1
    time_0 = time()
    for n_final_scen in n_final_scens:
        n_counter += 1
        f_counter = -1
        for fore in forecasters:
            f_counter +=1
            # battery sizing
            p_periods = []
            for k in np.arange(6):
                p_i = data_input['y_te'][k].T[0:96*n_days,0]
                p_periods.append(p_i)
            p_periods = np.hstack(p_periods)
            e_out_daily = np.abs(np.sum(p_periods[p_periods < 0]) * dt[0] / 3600 / n_days /6) # daily avarage produced energy in kWh
            c = 1
            cap = e_out_daily

            for k in np.arange(6):

                if fore=='rfhw':
                    # merge hw and rf forecasters

                    y_hat_hw = data_input['y_hat_hw'][k].T[0:96 * n_days, :] #* steps_size.reshape(1, -1)
                    y_hat_rf = data_input['y_hat'][k].T[0:96 * n_days, :] #* steps_size.reshape(1, -1)
                    y_i_hw = data_input['y_i_hw'][k].T[0:96 * n_days, :, :] #* steps_size.reshape(1, -1, 1)
                    y_i_hw = y_i_hw + 1e-6 * np.random.randn(y_i_hw.shape[0], y_i_hw.shape[1], y_i_hw.shape[2])
                    y_i_rf = data_input['y_i'][k].T[0:96 * n_days, :, :] #* steps_size.reshape(1, -1, 1)
                    y_i_rf = y_i_rf + 1e-6 * np.random.randn(y_i_rf.shape[0], y_i_rf.shape[1], y_i_rf.shape[2])
                    y_hat = y_hat_hw
                    y_hat[:,1:] = y_hat_rf[:,1:]
                    y_i = y_i_hw
                    y_i[:, 1:, :] = y_i_rf[:, 1:, :]
                else:
                    # rename stuff
                    y_hat = data_input['y_hat'+fore][k].T[0:96*n_days,:] #* steps_size.reshape(1,-1)
                    y_i = data_input['y_i'+fore][k].T[0:96*n_days,:,:]#* steps_size.reshape(1,-1,1)
                    y_i = y_i + 1e-6*np.random.randn(y_i.shape[0],y_i.shape[1],y_i.shape[2])

                y_te = data_input['y_te'][k].T[0:96 * n_days, :] #* steps_size.reshape(1, -1)
                data = {}
                data['y_te'] = y_te
                data['y_hat'] = y_hat
                data['scenarios'] = y_i
                data_pre = {}
                data_pre['y_te'] = y_te
                data_pre['y_hat'] = y_te
                data_pre['scenarios'] = y_i

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
                        'n_final_scens':n_final_scen,
                        'n_init_scens':n_init_scen}

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
                t0=time()
                history_stoc, cost_stoc, peak_sh_stoc, cost_real_stoc, peak_sh_real_stoc= stoc_batt.do_mpc(n_steps,P_obs,do_plots=False)
                dt_stoc = time()-t0
                t1=time()
                history_pre, cost_pre, peak_sh_pre, cost_real_pre, peak_sh_real_pre= pre_batt.do_mpc(n_steps, P_obs,do_plots=False)
                dt_pre = time() - t1
                history_det, cost_det, peak_sh_det, cost_real_det, peak_sh_real_det = det_batt.do_mpc(n_steps, P_obs,do_plots=False)

                '''
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
                plt.pause(2)
                '''
                # retrieve KPIs
                KPI_det[k,f_counter,n_counter]  = np.sum(cost_det + peak_sh_det)
                KPI_stoc[k,f_counter,n_counter] = np.sum(cost_stoc + peak_sh_stoc)
                KPI_pre[ k,f_counter,n_counter] = np.sum(cost_pre + peak_sh_pre)
                KPI_det_real[k,f_counter,n_counter]  = np.sum(cost_real_det + peak_sh_real_det)
                KPI_stoc_real[k,f_counter,n_counter] = np.sum(cost_real_stoc + peak_sh_real_stoc)
                KPI_pre_real[ k,f_counter,n_counter] = np.sum(cost_real_pre + peak_sh_real_pre)

                KPI_c_det[k,f_counter,n_counter]  = np.sum(cost_det)
                KPI_c_stoc[k,f_counter,n_counter] = np.sum(cost_stoc)
                KPI_c_pre[ k,f_counter,n_counter] = np.sum(cost_pre)
                KPI_c_det_real[k,f_counter,n_counter]  = np.sum(cost_real_det)
                KPI_c_stoc_real[k,f_counter,n_counter] = np.sum(cost_real_stoc)
                KPI_c_pre_real[ k,f_counter,n_counter] = np.sum(cost_real_pre)

                KPI_p_det[k,f_counter,n_counter]  = np.sum(peak_sh_det)
                KPI_p_stoc[k,f_counter,n_counter] = np.sum(peak_sh_stoc)
                KPI_p_pre[k,f_counter,n_counter] = np.sum(peak_sh_pre)
                KPI_p_det_real[k,f_counter,n_counter]  = np.sum(peak_sh_real_det)
                KPI_p_stoc_real[k,f_counter,n_counter] = np.sum(peak_sh_real_stoc)
                KPI_p_pre_real[k,f_counter,n_counter] = np.sum(peak_sh_real_pre)

                times_stoc[k,f_counter,n_counter] = dt_stoc
                times_pre[k,f_counter,n_counter] = dt_pre

                dtime = time()-t0
                print(dtime/60, c,cap,(KPI_det[k,f_counter,n_counter]-KPI_stoc[k,f_counter,n_counter])/KPI_stoc[k,f_counter,n_counter],
                      (KPI_det[k,f_counter,n_counter]-KPI_pre[k,f_counter,n_counter])/KPI_det[k,f_counter,n_counter])

                plt.close('all')

    results = {}
    results['KPT_det'] = KPI_det
    results['KPT_stoc'] = KPI_stoc
    results['KPT_pre'] = KPI_pre
    results['KPT_det_real'] = KPI_det_real
    results['KPT_stoc_real'] = KPI_stoc_real
    results['KPT_pre_real'] = KPI_pre_real

    results['KPT_c_det'] = KPI_c_det
    results['KPT_c_stoc'] = KPI_c_stoc
    results['KPT_c_pre'] = KPI_c_pre
    results['KPT_c_det_real'] = KPI_c_det_real
    results['KPT_c_stoc_real'] = KPI_c_stoc_real
    results['KPT_c_pre_real'] = KPI_c_pre_real

    results['KPT_p_det'] = KPI_p_det
    results['KPT_p_stoc'] = KPI_p_stoc
    results['KPT_p_pre'] = KPI_p_pre
    results['KPT_p_det_real'] = KPI_p_det_real
    results['KPT_p_stoc_real'] = KPI_p_stoc_real
    results['KPT_p_pre_real'] = KPI_p_pre_real

    results['dt_stoc'] = times_stoc
    results['dt_pre'] = times_pre
    total_time = time()-time_0
    print('total time: %0.2e' % total_time)
    return results


if __name__ == "__main__":
    numCPU = mp.cpu_count()
    print('starting parallel job on %d cores' % numCPU)

    jobs = np.linspace(0, 19,20, dtype=int)

    qw = []
    [qw.append(mp.Queue()) for j in jobs]
    qfr = mp.Queue()

    fr = mp.Process(target=fileReader, args=[qfr, qw])
    fr.daemon = True
    fr.start()

    p = mp.Pool(numCPU, init, [qfr, qw])

    results = p.map(do_something, jobs)

    np.save('results/sequential_results', results)

