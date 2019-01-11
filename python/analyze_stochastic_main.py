import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors

def quad_patch(lengths,centers):
    rect = []
    z = zip(lengths, centers)
    for l,c in z:
        rect.append(Rectangle((c-l/2,v-l/2),l,l))

    pc = PatchCollection(rect, facecolor='r', alpha=0.1,
                         edgecolor='b')
    return pc

scen_sets = ['10_20','20_30','30_40','30_60','30_80','40_80']
kpi_str = '\mathrm{KPI}^*'

c_max = 0
c_min = 10
cc_max = 0
cc_min = 10
cps_max = 0
cps_min = 10

for scen_set in scen_sets:

    results = np.load('results/results_'+scen_set + '.npy')
    results_names = results.flatten()[0].keys()
    data = {}
    for n in results_names:
        data[n] = results.flatten()[0][n]

    # data is formatted as [c,cap,fold,forecaster_type]
    n_i,n_j,n_f,n_fo = np.shape(data['KPT_det'])

    res_dict = {}
    forecasters_names = ['RF','Hw','ELM']
    fig1,ax2 = plt.subplots(1,3,figsize=(12, 4))
    cmap = plt.get_cmap('Set1')
    line_colors = cmap(np.linspace(0, 1, 6))

    # PLOT delta KPI, with respect to the forecaster deterministic score
    fig = plt.figure(figsize=(12, 4))
    gs = plt.GridSpec(3, 23)
    for fo in np.arange(n_fo):
        for k in results_names:
            res_dict[k] = np.mean(np.squeeze(data[k][:,:,:,fo]),2)

        K_det_norm = res_dict['KPT_det'] / res_dict['KPT_pre']
        K_sto_norm = res_dict['KPT_stoc'] / res_dict['KPT_pre']
        Kc_det_norm = res_dict['KPT_c_det'] / res_dict['KPT_c_pre']
        Kc_sto_norm = res_dict['KPT_c_stoc'] / res_dict['KPT_c_pre']
        Kps_det_norm = res_dict['KPT_p_det'] / res_dict['KPT_p_pre']
        Kps_sto_norm = res_dict['KPT_p_stoc'] / res_dict['KPT_p_pre']

        c_max_f = np.max([K_det_norm, K_sto_norm])
        c_min_f = np.min([K_det_norm, K_sto_norm])
        cc_max_f = np.max([Kc_det_norm, Kc_sto_norm])
        cc_min_f = np.min([Kc_det_norm, Kc_sto_norm])
        cps_max_f = np.max([Kps_det_norm, Kps_sto_norm])
        cps_min_f = np.min([Kps_det_norm, Kps_sto_norm])

        c_max = np.maximum(c_max, c_max_f)
        c_min = np.minimum(c_min, c_min_f)
        cc_max = np.maximum(cc_max, cc_max_f)
        cc_min = np.minimum(cc_min, cc_min_f)
        cps_max = np.maximum(cps_max, cps_max_f)
        cps_min = np.minimum(cps_min, cps_min_f)

    for fo in np.arange(n_fo):
        # obtain fold means
        #fig, ax = plt.subplots(1,6, figsize=(16, 3))

        window = np.arange(3)
        ax=[]
        ax.append(fig.add_subplot(gs[fo, 0:3]))
        ax.append(fig.add_subplot(gs[fo, 3:6]))
        ax.append(fig.add_subplot(gs[fo, 8:11]))
        ax.append(fig.add_subplot(gs[fo, 11:14]))
        ax.append(fig.add_subplot(gs[fo, 16:19]))
        ax.append(fig.add_subplot(gs[fo, 19:22]))

        x_ticklabels = np.array([0.5, 1, 1.5])
        y_ticklabels = np.array([0.5, 1, 1.5])

        for k in results_names:
            res_dict[k] = np.mean(np.squeeze(data[k][:,:,:,fo]),2)

        K_det_norm = res_dict['KPT_det'] / res_dict['KPT_pre']
        K_sto_norm = res_dict['KPT_stoc'] / res_dict['KPT_pre']
        Kc_det_norm = res_dict['KPT_c_det'] / res_dict['KPT_c_pre']
        Kc_sto_norm = res_dict['KPT_c_stoc'] / res_dict['KPT_c_pre']
        Kps_det_norm = res_dict['KPT_p_det'] / res_dict['KPT_p_pre']
        Kps_sto_norm = res_dict['KPT_p_stoc'] / res_dict['KPT_p_pre']

        '''
        c_max = np.max([K_det_norm, K_sto_norm])
        c_min = np.min([K_det_norm, K_sto_norm])
        cc_max = np.max([Kc_det_norm, Kc_sto_norm])
        cc_min = np.min([Kc_det_norm, Kc_sto_norm])
        cps_max = np.max([Kps_det_norm, Kps_sto_norm])
        cps_min= np.min([Kps_det_norm, Kps_sto_norm])
        '''

        cnorm = colors.PowerNorm(gamma=0.3)
        cnormc2 = colors.PowerNorm(gamma=0.8)

        cnormc = colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                  vmin=cc_min, vmax=cc_max)
        cnormps = colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                  vmin=cps_min, vmax=cps_max)
        colmap = cm.magma
        im1 = ax[0].imshow(K_det_norm, clim=(c_min,c_max),cmap=colmap,norm=cnorm)
        im2 = ax[1].imshow(K_sto_norm, clim=(c_min,c_max),cmap=colmap,norm=cnorm)
        im3 = ax[2].imshow(Kc_det_norm, clim=(cc_min,cc_max),cmap=colmap,norm=cnormc2)
        im4 = ax[3].imshow(Kc_sto_norm, clim=(cc_min,cc_max),cmap=colmap,norm=cnormc2)
        im5 = ax[4].imshow(Kps_det_norm, clim=(cps_min,cps_max),cmap=colmap,norm=cnorm)
        im6 = ax[5].imshow(Kps_sto_norm, clim=(cps_min,cps_max),cmap=colmap,norm=cnorm)
        print(res_dict['KPT_pre'][-1,-1])


        # add colorbar
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.2, hspace=0.001)

        '''
        position = ax[1].get_position().get_points()
        cb_ax1 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, position[1][1] - position[0][1]])
        position = ax[3].get_position().get_points()
        cb_ax2 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, position[1][1] - position[0][1]])
        position = ax[5].get_position().get_points()
        cb_ax3 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, position[1][1] - position[0][1]])
        cbar1 = fig.colorbar(im1, cax=cb_ax1)
        cbar2 = fig.colorbar(im3, cax=cb_ax2)
        cbar3= fig.colorbar(im5, cax=cb_ax3)
        '''
        ax[1].get_yaxis().set_ticks([])
        ax[2].get_yaxis().set_ticks([])
        ax[3].get_yaxis().set_ticks([])
        ax[4].get_yaxis().set_ticks([])
        ax[5].get_yaxis().set_ticks([])
        ax[0].get_yaxis().set_ticks(np.arange(res_dict['KPT_det'].shape[0]))
        ax[0].get_yaxis().set_ticklabels(y_ticklabels)
        ax[0].set_ylabel(forecasters_names[fo])

        x_label = r'$E_{nom}^*$ $[-]$'
        x_label = None
        for fr in np.arange(6):
            ax[fr].get_xaxis().set_ticks(np.arange(res_dict['KPT_det'].shape[1]))
            ax[fr].get_xaxis().set_ticklabels(x_ticklabels)
            ax[fr].set_xlabel(x_label)

        if fo<n_fo-1:
            for fr in np.arange(6):
                ax[fr].get_xaxis().set_ticklabels([])
                ax[fr].set_xlabel([])

        if fo==0:
            ax[0].set_title(r'$'+kpi_str+'$ det ' )
            ax[1].set_title(r'$'+kpi_str+'$ stoc ')

            ax[2].set_title(r'$'+kpi_str+'_c$ det ' )
            ax[3].set_title(r'$'+kpi_str+'_c$ stoc ' )

            ax[4].set_title(r'$'+kpi_str+'_{ps}$ det ')
            ax[5].set_title(r'$'+kpi_str+'_{ps}$ stoc ')


        #plt.savefig('results/rel_diff'+forecasters_names[fo]+ scen_set +'.pdf',format='pdf')

        # PLOT delta KPI, with respect to the forecaster deterministic score

        if fo<2:
            K_det_norm = res_dict['KPT_det'] / (res_dict['KPT_pre'])
            K_sto_norm = res_dict['KPT_stoc'] /( res_dict['KPT_pre'])
            Kc_det_norm = res_dict['KPT_c_det'] / (res_dict['KPT_c_pre'])
            Kc_sto_norm = res_dict['KPT_c_stoc'] / (res_dict['KPT_c_pre'])
            Kps_det_norm = res_dict['KPT_p_det'] / (res_dict['KPT_p_pre'])
            Kps_sto_norm = res_dict['KPT_p_stoc'] / (res_dict['KPT_p_pre'])
            ax2[0].plot(K_det_norm.ravel(), '.', color=line_colors[fo, :],label = 'det, '+forecasters_names[fo])
            ax2[0].plot(K_sto_norm.ravel(), '+', color=line_colors[fo, :],label = 'stoc, '+forecasters_names[fo])
            ax2[1].plot(Kc_det_norm.ravel(), '.', color=line_colors[fo, :])
            ax2[1].plot(Kc_sto_norm.ravel(), '+', color=line_colors[fo, :])
            ax2[2].plot(Kps_det_norm.ravel(), '.', color=line_colors[fo, :])
            ax2[2].plot(Kps_sto_norm.ravel(), '+', color=line_colors[fo, :])
            ax2[0].set_ylim(0.98,1.21)
            ax2[1].set_ylim(0.98,1.21)
            ax2[2].set_ylim(0.98,1.21)
            ax2[0].legend()

    position = ax[1].get_position().get_points()
    cb_ax1 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, 3*(position[1][1] - position[0][1])])
    position = ax[3].get_position().get_points()
    cb_ax2 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, 3*(position[1][1] - position[0][1])])
    position = ax[5].get_position().get_points()
    cb_ax3 = fig.add_axes([position[1][0] + 0.01, position[0][1], 0.02, 3*(position[1][1] - position[0][1])])
    cbar1 = fig.colorbar(im1, cax=cb_ax1)
    cbar2 = fig.colorbar(im3, cax=cb_ax2)
    cbar3 = fig.colorbar(im5, cax=cb_ax3)

    x_ticklabels = ['0.5\n0.5','0.5\n1','0.5\n1.5','1\n0.5','1\n1','1\n1.5','1.5\n0.5','1.5\n1','1.5\n1.5']
    ax2[1].get_yaxis().set_ticklabels([])
    ax2[2].get_yaxis().set_ticklabels([])
    ax2[0].get_xaxis().set_ticks(np.arange(9))
    ax2[1].get_xaxis().set_ticks(np.arange(9))
    ax2[2].get_xaxis().set_ticks(np.arange(9))
    ax2[0].get_xaxis().set_ticklabels(x_ticklabels)
    ax2[1].get_xaxis().set_ticklabels(x_ticklabels)
    ax2[2].get_xaxis().set_ticklabels(x_ticklabels)

    ax2[0].set_title(r'$'+kpi_str+'$')
    ax2[1].set_title(r'$'+kpi_str+'_c$')
    ax2[2].set_title(r'$'+kpi_str+'_{ps}$')
    ax2[0].grid(True)
    ax2[1].grid(True)
    ax2[2].grid(True)

    plt.pause(0.1)
    fig1.suptitle(scen_set)
    plt.show()
    fig1.savefig('results/improvement' + scen_set + '.pdf', format='pdf')
    fig.savefig('results/heatmaps' + scen_set + '.pdf', format='pdf')

# plot times
t_stoc = []
t_det = []
for scen_set in scen_sets:

    results = np.load('results/results_'+scen_set + '.npy')
    results_names = results.flatten()[0].keys()
    data = {}
    for n in results_names:
        data[n] = results.flatten()[0][n]

    t_stoc.append(data['dt_stoc'].flatten()/96/2)
    t_det.append(data['dt_pre'].flatten()/96/2)

t_det = [np.hstack(t_det).flatten()]
fig,ax = plt.subplots(1)
ax.boxplot(t_det + t_stoc,notch=True,sym='+',patch_artist=True)
ax.set_yscale('log')
ax.set_yticks(10.**np.arange(-2, 2))
ax.set_yticklabels(10.0**np.arange(-2, 2))
S = [s.split('_') for s in scen_sets]
x_ticklabels = ['det']
for s in S:
    s = s[0]+'\n'+s[1]
    x_ticklabels.append(s)
ax.get_xaxis().set_ticklabels(x_ticklabels)
plt.grid(True,'minor',linestyle='--')
plt.ylabel('CPU time [s]')
plt.show()
fig.savefig('results/times.pdf', format='pdf')

#plt.close('all')