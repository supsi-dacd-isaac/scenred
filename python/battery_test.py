from forecasters import RELM
import scipy.io as io
from scenred import scenred, plot_scen, plot_graph
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from _battery_controller import BatteryController
import networkx as nx
import battery


np.random.seed(0)
# create an easy signal
n_days = 100
obs_per_day = 15
t = np.linspace(0,n_days*obs_per_day,n_days*obs_per_day)
N = len(t)
y = np.sin(t*n_days*2*np.pi/N)
x_cos = np.cos(t*n_days/N)+np.random.rand(N)
y_past = y +np.random.rand(N)
# henkelize
X = []
target = []
for e in np.arange(2*obs_per_day):
    win = e+np.linspace(0,N-2*obs_per_day-1,N-2*obs_per_day,dtype=int)
    X.append(np.hstack([x_cos[win].reshape(-1,1),y_past[win].reshape(-1,1)]))
    target.append(y[win].reshape(-1,1))

dataset = {}
X = np.hstack(X)
target = np.hstack(target)

# online forecasting
dataset['X_tr'] = X[0:int(N*0.8),:]
dataset['y_tr']= target[0:int(N*0.8),:]
dataset['X_te']= X[int(N*0.8):,:]
dataset['y_te'] = target[int(N*0.8):,:]


dt = 60*15
pars = {'h':30,
        'ts':np.atleast_1d(dt),
        'ps':0.07,
        'pb':0.2,
        'type':'stochastic',
        'alpha':1,
        'rho':1,
        'n_final_scens':10,
        'n_init_scens':5}


batt = battery.Battery(dataset=dataset,pars=pars,c_nom=1,cap_nom=5)
P_final, P_controlled, P_uncontrolled, U, SOC = batt.solve_step(time=0)


plt.figure()
U = batt.battery_controller.u_st.value
for s in np.arange(batt.battery_controller.scen_idxs.shape[1]):
    u_s = U[batt.battery_controller.scen_idxs[:,s].ravel(),:]
    plt.plot(u_s[:, 0],color='b',alpha=0.2,linewidth=0.5)
    plt.plot(u_s[:, 1],color='r',alpha=0.2,linewidth=0.5)

plt.figure()
PS =P_uncontrolled
plt.plot(P_final, '.', color='k', alpha=0.2, linewidth=0.5)
for s in np.arange(batt.battery_controller.scen_idxs.shape[1]):
    u_s = U[batt.battery_controller.scen_idxs[:,s].ravel(),:]
    Pms = PS[batt.battery_controller.scen_idxs[:,s].ravel()]
    plt.plot(Pms +u_s[:, [0]]-u_s[:, [1]],color='b',alpha=0.2,linewidth=0.5)
    plt.plot(Pms,color='r',alpha=0.2,linewidth=0.5)

N = 10
batt.online_stochastic_plot(N)

