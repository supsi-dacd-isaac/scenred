import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

results = np.load('results.npy')
results_names = results.flatten()[0].keys()
data = {}
for n in results_names:
    data[n] = results.flatten()[0][n]

# data is formatted as [c,cap,fold,forecaster_type]
n_i,n_j,n_f,n_fo = np.shape(data['KPT_det'])

res_dict = {}
fig,ax = plt.subplots(4,3,figsize = (9,12))
x_ticklabels = np.linspace(0.5, 2, 4)
for fo in np.arange(n_fo):
    # obtain fold means
    for k in results_names:
        res_dict[k] = np.mean(np.squeeze(data[k][:,:,:,fo]),2)

    c_max = np.max([res_dict['KPT_det'],res_dict['KPT_stoc'],res_dict['KPT_pre']])
    c_min = np.min([res_dict['KPT_det'], res_dict['KPT_stoc'], res_dict['KPT_pre']])

    im1 = ax[fo, 0].imshow(res_dict['KPT_det'], clim=(c_min,c_max))
    im2 = ax[fo, 1].imshow(res_dict['KPT_stoc'], clim=(c_min,c_max))
    im3 = ax[fo, 2].imshow(res_dict['KPT_pre'], clim=(c_min,c_max))
    ax[fo,1].get_yaxis().set_ticks([])
    ax[fo,2].get_yaxis().set_ticks([])
    if not fo==n_fo-1:
        ax[fo, 0].get_xaxis().set_ticks([])
        ax[fo, 1].get_xaxis().set_ticks([])
        ax[fo, 2].get_xaxis().set_ticks([])
    else:
        ax[fo, 0].get_xaxis().set_ticks(np.arange(res_dict['KPT_det'].shape[1]))
        ax[fo, 1].get_xaxis().set_ticks(np.arange(res_dict['KPT_det'].shape[1]))
        ax[fo, 2].get_xaxis().set_ticks(np.arange(res_dict['KPT_det'].shape[1]))

        ax[fo, 0].get_xaxis().set_ticklabels(x_ticklabels)
        ax[fo, 1].get_xaxis().set_ticklabels(x_ticklabels)
        ax[fo, 2].get_xaxis().set_ticklabels(x_ticklabels)


position = ax[fo,2].get_position().get_points()
cb_ax = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, position[1][1] - position[0][1]])
cbar = fig.colorbar(im3, cax=cb_ax)
