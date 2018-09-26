import scipy.io as io
import matplotlib.pyplot as plt
from scenred import scenred
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

# specify accuracy
[S_tol,P_tol,J_tol,Me_tol] = scenred(np.copy(data), metric = 'cityblock',tol = 0.1)
fig =  plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(S_tol.shape[1]):
    ax.plot(np.arange(S_tol.shape[0]),np.squeeze(S_tol[:,i,0]),np.squeeze(S_tol[:,i,1]),color='k',alpha=0.01)

# specify scenarios
[S_s,P_s,J_s,Me_s] = scenred(np.copy(data), metric = 'cityblock',nodes=np.linspace(1,30,T.shape[0],dtype=int))
fig =  plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(S_s.shape[1]):
    ax.plot(np.arange(S_s.shape[0]),np.squeeze(S_s[:,i,0]),np.squeeze(S_s[:,i,1]),color='k',alpha=0.1)




