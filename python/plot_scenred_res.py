import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import h5py
from forecasters import pre_trained_forecaster
from scenred import plot_graph
from scenred import plot_scen
from matplotlib import cm
np.random.seed(0)

filename = '/home/queen/Documents/ISAAC_matlab_svn/BB_forecasting/hierarchical_forecast/Data/base_forecasters_control.mat'
f = h5py.File(filename, 'r')

n_days = 10
k = 0
n = 1
y_hat_hw = f[f['bf']['y_hat_hw'][k][n]].value.T[0:96 * n_days, :]  # * steps_size.reshape(1, -1)
y_hat_rf = f[f['bf']['y_hat'][k][n]].value.T[0:96 * n_days, :]  # * steps_size.reshape(1, -1)
y_i_hw = f[f['bf']['y_i_hw'][k][n]].value.T[0:96 * n_days, :, :]  # * steps_size.reshape(1, -1, 1)
y_i_hw = y_i_hw + 1e-6 * np.random.randn(y_i_hw.shape[0], y_i_hw.shape[1], y_i_hw.shape[2])
y_hat = y_hat_hw
y_i = y_i_hw
#y_i[:, 1:, :] = y_i_rf[:, 1:, :]
y_te = f[f['bf']['y_te'][k][n]].value.T[0:96 * n_days, :]  # * steps_size.reshape(1, -1)

dataset = {}
dataset['y_hat'] = y_hat
dataset['scenarios'] = y_i
scenarios_per_step = np.linspace(1,10,10)
ptf = pre_trained_forecaster(dataset,scenarios_per_step)

t = 82 # time
g, S_s, S_init = ptf.predict_scenarios(t)

fig, ax = plt.subplots(1,2,figsize=(12, 5))
axg,cb = plot_graph(g,ax[1])

axg.set_title('Scenario DAG')
cb.set_label('Power [kW]')
axg.grid(True)

plt.savefig('results/DAG.pdf',format='pdf')

S_s[0,:,0] = np.mean(y_i[t,0,:])
cmap = plt.get_cmap('plasma')
line_colors = cmap(np.linspace(0, 1, 2))

steps_size = np.cumsum(np.array([1,1,2,3,5,7,10,15,21,31])*15)
im1 = ax[0].plot(steps_size,y_i[t,:,:],alpha = 0.1,linewidth=0.5,color=line_colors[0,:])
ax[0].plot(steps_size,np.squeeze(S_s) ,color='k', alpha=0.7,marker='.')
ax[0].set_xscale('log')
ax[0].set_ylabel('Power [kW]')
ax[0].set_xlabel('time ahead [min]')


plt.show()
plt.savefig('results/scenario_tree.pdf',format='pdf')

plt.close('all')


