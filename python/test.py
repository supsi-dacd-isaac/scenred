import scipy.io as io
from scenred import scenred, plot_scen, plot_graph
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

mat = io.loadmat('../data.mat')
V = mat['data'][0,0]
T = mat['data'][0,1]

# plot some of the scenarios
fig,axes = plt.subplots(2,1)
axes[0].plot(V[:,0:100])
axes[1].plot(T[:,0:100])

data = np.zeros((np.shape(T)[0],np.shape(T)[1],2))
data[:,:,0] = V
data[:,:,1] = T

# ---------- Multivariate scenarios, eps method -----------------------------------------------------
# specify accuracy. This will raise a warning due to the dimension of the identified tree, saying you have to manually
# call the get_netork() function if you really want to retrieve it. g is an empty element
[S_tol,P_tol,J_tol,Me_tol,g] = scenred(np.copy(data), metric = 'cityblock',tol = 0.1)

# plot the scenarios
plot_scen(S_tol)

# ---------- Multivariate scenarios, num scen method -----------------------------------------------------
# identify the scenario tree specifying the scenario number
[S_s,P_s,J_s,Me_s,g] = scenred(np.copy(data), metric = 'cityblock',nodes=np.linspace(1,30,T.shape[0],dtype=int))

# plot the scenarios
plot_scen(S_s)
plot_graph(g)

#
plt.figure()
plt.spy(Me_s[10,:,:])
plt.title('scenario link structure, time = %i' % 10)

# ---------- Univariate scenarios, num scen method -----------------------------------------------------
# specify scenarios, only T
data_T = data[:,:,[1]]
[S_s,P_s,J_s,Me_s,g] = scenred(np.copy(data_T), metric = 'cityblock',nodes=np.linspace(1,30,T.shape[0],dtype=int))
# plot the scenarios
plot_scen(S_s)
plot_graph(g)

