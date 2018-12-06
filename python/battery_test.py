from forecasters import RELM
import scipy.io as io
from scenred import scenred, plot_scen, plot_graph
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from _battery_controller import BatteryController
import networkx as nx

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

X = np.hstack(X)
target = np.hstack(target)
X_tr = X[0:int(N*0.8),:]
target_tr = target[0:int(N*0.8),:]
X_te = X[int(N*0.8):,:]
target_te = target[int(N*0.8):,:]


# create predictor
N_FINAL_SCENARIOS = 10
scens_per_step = np.linspace(5,N_FINAL_SCENARIOS ,target_te.shape[1],dtype=int)
relm =  RELM(scenarios_per_step=scens_per_step,lamb=1e-5)
relm.train(X_tr,target_tr)
y_hat,quantiles,y_i = relm.predict(X_te)

plt.figure()
plt.plot(target_te[:,0])
plt.plot(y_hat[:,0])
plt.plot(np.squeeze(y_i[:,0,:]),linewidth=0.1)

g,S_s = relm.predict_scenarios(X_te[[0],:])
plot_scen(S_s,target_te[0,:])
plot_graph(g)

# ----- Instantiate a battery controller ----------

ts = 60*10 # ten minutes sampling time
h = 144
h_mult = S_s.shape[0]

# set random seed
np.random.seed(0)
sum_stp = 0
expend = 1
# use logarithmic spacing for the aggregation
while sum_stp != h:
    tsteps = np.asanyarray(np.logspace(0,expend,h_mult),int)
    sum_stp = np.sum(tsteps)
    expend = expend+0.0001
    print('%f' % sum_stp)

ts_mult = ts*tsteps

alpha = 1
rho = 0.1
k = alpha / (2 * rho)
controller_pars = {'e_n': 1,
                   'c': 1/2,
                   'ts': ts_mult,
                   'tau_sd': 365*24*3600,
                   'lifetime': 20,
                   'dod': 0.8,
                   'nc': 3000,
                   'eta_in': 0.9,
                   'eta_out': 0.9,
                   'h': h_mult,
                   'pb': 0.2,
                   'ps': 0.07,
                   'type': 'stochastic',
                   'alpha': alpha,
                   'rho': rho,
                   'g':g}
batt_cont = BatteryController(pars=controller_pars)
#batt_cont.solve_step(np.array(list(nx.get_node_attributes(g,'v').values())[0:-1] ).reshape(-1,1))
ppp  = -np.array(list(nx.get_node_attributes(g, 'v').values()))
batt_cont.solve_step(g)

plt.figure()
U = batt_cont.u_st.value
for s in np.arange(batt_cont.scen_idxs.shape[1]):
    u_s = U[batt_cont.scen_idxs[:,s].ravel(),:]
    plt.plot(u_s[:, 0],color='b',alpha=0.2,linewidth=0.5)
    plt.plot(u_s[:, 1],color='r',alpha=0.2,linewidth=0.5)

plt.figure()
PS = np.array(list(nx.get_node_attributes(g,'v').values()))
for s in np.arange(batt_cont.scen_idxs.shape[1]):
    u_s = U[batt_cont.scen_idxs[:,s].ravel(),:]
    Pms = PS[batt_cont.scen_idxs[:,s].ravel()]
    plt.plot(Pms +u_s[:, [0]]-u_s[:, [1]],color='b',alpha=0.2,linewidth=0.5)
    plt.plot(Pms,color='r',alpha=0.2,linewidth=0.5)
