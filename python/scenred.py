import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import warnings
from itertools import count
import copy

def get_dist(X,metric):
    D = squareform(pdist(X,metric))
    return D

def scenred(samples, **kwargs):
    '''
    Build a scenario tree reducing N observed samplings from a multivariate
    distribution. The algorithm is described in "Nicole Growe-Kuska,
    Holger Heitsch and Werner Romisch - Scenario Reduction Scenario Tree
    Construction for Power Management Problems".
     Inputs:
             samples: n cell array, where n is the number of signals observed,
             each cell of dimension T*n_obs, where T is the number of observed
             timesteps and n_obs is the original number of observed scenarios.
             The scenarios in the different cells are supposed to be
             statistically dependent (sampled from historical data or from
             a copula)
             metric: in {'euclidean','cityblock'}, used for
                     Kantorovich-Rubinstein metric
             kwargs: {'nodes','tol'} methods for building the tree
                       -nodes vector with length T, increasingly monotone,
                       with the specified number of nodes per timestep
                       -tol: tolerance on Dist/Dist_0, where Dist is the
                             distance after aggregating scenarios, and Dist_0
                             is the distance between all the scenarios and a
                             tree with only one scenario. In any case, at
                             least one scenario is selected.
     Outputs:
              S: T*n_obs*n matrix, containing all the path in the scenario
              tree
              P: T*n_obs matrix, containing evolviong probabilities for each
              scenario
              J: T*n_obs matrix, indicating non-zero entries of P (if a
              scenario is present at a given timestep)
              Me: T*n_scen*n_scen matrix, containing which scenario is linked
              with which at time t
    :return:
    '''

    T = samples.shape[0]
    n_obs = samples.shape[1]

    defaultNodes = np.ones((T,1))
    # pars kwargs
    pars = {'nodes':defaultNodes,
            'tol': 10,
            'metric':'cityblock'}
    for key in ('nodes','tol','metric'):
        if key in kwargs:
            pars[key] = kwargs[key]
    # Obtain the observation matrix, size (n*T) x n_obs, from which to compute distances. Normalize observations
    X = []
    S = samples
    for i in np.arange(np.shape(S)[2]):
        V = S[:,:,i]
        V_norm = (V-np.mean(V,1).reshape(-1,1))/(np.std(V,1)+1e-6).reshape(-1,1)
        X.append(V_norm)

    X=np.vstack(X).T

    D = get_dist(X, pars['metric'])
    D = D + np.eye(D.shape[0]) * (1 + np.max(D.ravel()))
    infty = 1e12

    # generate the tolerance vector
    if all(pars['nodes'].ravel() == defaultNodes.ravel()):
        #Tol = np.tile(pars['tol'], (1, T))[0]
        Tol = np.fliplr(pars['tol'] / (1.5 ** (T - np.arange(T).reshape(1,-1)+1))).ravel()
        Tol[0] = infty
    else:
        Tol = infty * np.ones((1, T)).ravel()

    if pars['nodes'][0] != 1:
        print('The first number of scenarios is not 1. This will cause an error on graph retrieval. I am forcing it to 1')
        pars['nodes'][0] = 1

    J = np.asanyarray(np.ones((T,n_obs)),bool)
    L = np.zeros((n_obs,n_obs))
    P = np.ones((T, n_obs)) / n_obs
    branches = n_obs
    for i in np.fliplr(np.arange(T).reshape(1,-1))[0]:
        delta_rel = 0
        delta_p = 0
        D_i = D

        basic_idx = np.asanyarray(np.hstack([np.ones((1, i+1)), np.zeros((1, T - i-1))]),bool)
        sel_idx = np.tile(basic_idx, (1, samples.shape[2])).ravel()
        X_filt = X[J[i,:], :]
        X_filt = X_filt[:, sel_idx.ravel()]
        D_j = get_dist(X_filt, pars['metric'])
        D_j[np.asanyarray(np.eye(np.shape(D_j)[0]),bool)] = 0 # do not consider self - distance
        delta_max = np.min(np.sum(D_j,0))

        while (delta_rel < Tol[i]) and (branches > pars['nodes'][i]):
            D_i[~J[i,:],:] =  infty  # set distance of discarded scenarios to infinity, ignoring them
            D_i[:, ~J[i,:]] = infty
            d_s = np.sort(D_i,0)  # sort distances with respect to the non-discarded scenarios
            idx_s = np.argsort(D_i,0)
            z = d_s[0,:]*P[i,:]  # vector of weighted probabilities
            #z[z == 0] = infty  # set prob. of removed scenario to inf in order to ignore them
            z[~J[i,:]] = infty  # set prob. of removed scenario to inf in order to ignore them
            idx_rem = np.argmin(z)  # find the scenario which cause the smallest p*d-deviation when merged, and its index
            dp_min = np.min(z)
            idx_aug = idx_s[0, idx_rem]  # retrieve who's being augmented with the probability of idx_rem
            J[i, idx_rem] = False  # mark it as a removed scenarion in the current timestep
            P[i, idx_aug] = P[i, idx_rem] + P[i,idx_aug] # add the probability of the removed scenario to the closest scenario
            P[i, idx_rem] = 0  # set probability of removed scenarios to 0
            branches = np.sum(P[i,:] > 0)  # count remaining branches
            L[idx_aug, idx_rem] = 1  # keep track of which scenario has merged
            L[idx_aug, L[idx_rem,:] > 0] = 1  # the scenario who's been augmented heredit all the scenarios previously merged with the removed scenario
            S[0: i+1, idx_rem,:] =  S[0: i+1, idx_aug,:]  # make the merged scenarios equal up to the root node
            to_merge_idx = np.argwhere(L[[idx_rem],:])  # make all the scenario previously merged with the removed one equal to the one in idx_aug, up to the root node
            for j in np.arange(np.shape(to_merge_idx)[0]):
                S[: i+1, to_merge_idx[j,1],:] =  S[: i+1, idx_aug,:]

            # update the differential accuracy
            if not Tol[i]==infty:
                delta_p = delta_p + dp_min
            delta_rel = delta_p / delta_max

        if i > 0:
            # Update available scenarios in the previous timestep
            J[i - 1,:] = J[i,:]

            # update previous timestep probabilities
            P[i - 1,:] = P[i,:]
            D[~J[i,:], ~J[i,:]] = infty

        #print('Branches t=%i: %i' %  (i, branches))

    S = S[:, J[-1,:] > 0,:]
    P = P[:, J[-1,:] > 0]

    # obtain the match matrix
    Me = np.zeros((S.shape[1],S.shape[1],T))
    for i in np.arange(T):
        D = get_dist(S[i, :, 0].reshape(-1, 1), pars['metric'])
        #D[np.eye(D.shape[0],dtype=bool)] = 1
        Me[:,:,i] = D==0

    Me = np.swapaxes(Me,2,0)

    if np.shape(S)[1]<100:
        g = get_network(S, P)
    else:
        print('Warning: automatic retrieval of the network graph has been disabled because the number of '
                         'nodes is high. If you want to get the network graph anyway, call the get_network(S,P)')
        scenario_warning('Warning: automatic retrieval of the network graph has been disabled because the number of '
                         'nodes is high. If you want to get the network graph anyway, call the get_network(S,P)')
        g = []

    return S, P, J, Me, g

def plot_scen(S_s,y=None):
    '''

    :param S_s:
    :return:
    '''
    if S_s.shape[2]==2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in np.arange(S_s.shape[1]):
            ax.plot(np.arange(S_s.shape[0]), np.squeeze(S_s[:, i, 0]), np.squeeze(S_s[:, i, 1]), color='k', alpha=0.1)
        if y is not None:
            ax.plot(np.arange(S_s.shape[0]),y[:,0],y[:,1],linewidth=1.5)
    elif S_s.shape[2]==1:
        fig,ax = plt.subplots(1)
        plt.plot(np.squeeze(S_s),color='k', alpha=0.1)
        if y is not None:
            plt.plot(y, linewidth=1.5)
    else:
        assert S_s.shape[2]>2, 'Error: cannot visualize more than bivariate scenarios'
    return fig,ax

def plot_graph(g,ax=None):
    '''
    Plot the networkx graph which encodes the scenario tree
    :param g: the networkx graph which encodes the scenario tree
    :return:
    '''

    # get unique groups
    if ax is None:
        fig= plt.figure()
        ax =plt.gca()
    groups = set(np.array(list(nx.get_node_attributes(g, 'v').values()))[:, 0])
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    colors = [g.node[n]['v'][0] for n in nodes]
    p = np.array(list(nx.get_node_attributes(g, 'p').values()))

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = graphviz_layout(g, prog='dot')
    # nx.draw_networkx(g,pos,with_labels=True)
    ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors,
                                with_labels=True, node_size=100*p, cmap=plt.cm.magma)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(nc)
    return ax,cb

def plot_from_graph(g):
    f = plt.figure()
    s_idx, leaves = retrieve_scenarios_indexes(g)
    values = np.array(list(nx.get_node_attributes(g, 'v').values()))
    cmap = plt.get_cmap('Set1')
    line_colors = cmap(np.arange(3))
    for s in np.arange(s_idx.shape[1]):
        plt.plot(values[s_idx[:,s]],color=line_colors[0,:],linewidth=1,alpha = 1)

def get_network(S_s,P_s):
    '''
    Get a network representation from the S_s and P_p matrices. The network is encoded in a networkx graph, each node
    has the following attribute:
    t: time of the node
    p: probability of the node
    v: array of values associated with the node

    :param S_s: T*n_obs*n matrix, containing all the path in the scenario
                tree
    :param P_s: T*n_obs matrix, containing evolviong probabilities for each
                scenario
    :return: g: a networkx graph with time, probability and values encoded with the connectivity of the nodes
    '''

    g = nx.DiGraph()
    g.add_node(0, t=0, p=1, v=S_s[0, P_s[0, :] > 0, :].ravel())

    for t in 1 + np.arange(P_s.shape[0] - 1):
        for s in np.arange(P_s.shape[1]):
            # span all the times, starting from second point (the root node is already defined)
            # consider current point mu, and its previous point in its scenario
            mu = S_s[t, s, :]
            p = P_s[t, s]
            # if probability of current point is zero, just go on
            if p == 0:
                continue
            mu_past = S_s[t - 1, s, :]
            # get current times in the tree
            times = np.array(list(nx.get_node_attributes(g, 't').values()))
            # get values, filtered by current time
            values = np.array(list(nx.get_node_attributes(g, 'v').values()))
            values_t = values[times == t, :]
            values_past = values[times == t - 1, :]
            # check if values of current point s are present in the tree at current time
            if np.any([np.array_equal(mu, a) for a in values_t]):
                continue
            else:
                # find parent node
                try:
                    parent_node = [x for x, y in g.nodes(data=True) if y['t'] == t - 1 and np.array_equal(y['v'], mu_past)]
                except:
                    print('The tree does not have a root! It is likely that you did required more than one node as first node')
                # create the node and add an edge from current point to its parent
                new_node = len(g.nodes())
                g.add_node(new_node, t=t, p=p, v=mu,label=new_node)
                g.add_edge(parent_node[0], new_node)
                #print('node %i added, with values' % (new_node), mu)

    # approximated probabilities
    for i in np.arange(len(g.nodes)):
        g.nodes[i]['p2'] = np.sum([ g.nodes[n]['t']==t for n in  nx.descendants(g,i)])/np.sum([ g.nodes[n]['t']==t for n in  nx.descendants(g,0)])
        if g.nodes[i]['p2']==0:
            g.nodes[i]['p2'] = 1/len(nx.descendants(g,0))


    return g

def scenario_warning(message):
    warnings.warn(message)

def set_distance(alphas,samples,metric):
    D = np.zeros((samples.shape[0],alphas.shape[0]))
    for i in np.arange(alphas.shape[0]):
        D[:,[i]] = cdist(samples,alphas[[i],:],metric=metric)
    #d = np.sum(np.minimum(D,0))
    d = np.sum(np.min(D, 1))
    return d,D

def retrieve_scenarios_indexes(g):
    n_n = len(g.nodes)
    node_set = np.linspace(0,n_n-1,n_n,dtype=int)
    all_t = np.array(list(nx.get_node_attributes(g, 't').values()))
    t = np.unique(all_t)
    leafs = np.array([n for n in node_set[all_t == np.max(t)]])
    scen_idxs_hist = np.zeros((max(t)+1,len(leafs)),dtype=int)
    for s in np.arange(len(leafs)):
        scen_idxs = np.sort(np.array(list(nx.ancestors(g, leafs[s]))))
        scen_idxs = np.asanyarray(np.insert(scen_idxs, len(scen_idxs),leafs[s],0),int)
        scen_idxs_hist[:,s] = scen_idxs
    return scen_idxs_hist,leafs

def refine_scenarios(g,samples,metric):
    n_random = 20
    times = np.array(list(nx.get_node_attributes(g, 't').values()))
    # get values, filtered by current time
    values = np.array(list(nx.get_node_attributes(g, 'v').values()))
    d_history = []
    g_new = copy.deepcopy(g)
    for t in np.arange(len(np.unique(times))):
        alphas_t = values[times==t]
        node_list = np.array(list(g.nodes))[times==t]
        samples_t = samples[t,:,:]
        d,Dummy = set_distance(alphas_t,samples_t,metric)
        d_history.append(d)
        if t==0:
            alphas_t = np.atleast_2d(np.median(samples_t,0))
        else:
            D = squareform(pdist(alphas_t))
            D = D+np.eye(D.shape[0]) * (1 + np.max(D.ravel()))
            k_renorm = np.mean(np.min(D,0))/2
            # for all the alpha points
            for i in np.arange(alphas_t.shape[0]):
                new_alphas = alphas_t[i,:] + k_renorm*np.random.randn(n_random,alphas_t.shape[1])
                # for all the new candidates of each alpha point
                for j in np.arange(new_alphas.shape[0]):
                    alphas_test_set = np.copy(alphas_t)
                    alphas_test_set[i] = new_alphas[j]
                    d_new,Dummy = set_distance(alphas_test_set, samples_t,metric)
                    if d_new<d:
                        alphas_t[i] = new_alphas[j]
                        d = d_new
        d,D = set_distance(alphas_t, samples_t,metric)
        p_vect = np.bincount(np.argmin(D,1),np.ones(D.shape[0],dtype=int))/D.shape[0]

        for i in np.arange(len(node_list)):
            g_new.nodes[node_list[i]]['v'] = np.array(alphas_t[i,:])
            #g_new.nodes[node_list[i]]['p'] = np.array(p_vect[i])

    d_history_new = []
    values = np.array(list(nx.get_node_attributes(g_new, 'v').values()))
    for t in np.arange(len(np.unique(times))):
        alphas_t = np.array(values[times == t]).reshape(-1,np.size(values[0]))
        samples_t = samples[t, :, :]
        d,Dummy = set_distance(alphas_t, samples_t,metric)
        d_history_new.append(d)
    d_history = np.array(d_history).reshape(-1,1)
    d_history_new = np.array(d_history_new).reshape(-1, 1)

    # total distance, before and after
    distances = np.hstack([d_history, d_history_new])

    '''
    s_idx,leaves = retrieve_scenarios_indexes(g)
    s_idx_new, leaves_new = retrieve_scenarios_indexes(g_new)
    p = np.array(list(nx.get_node_attributes(g, 'p').values()))
    p_new = np.array(list(nx.get_node_attributes(g_new, 'p').values()))
    for s in np.arange(s_idx.shape[1]):
        plt.plot(p[s_idx[:,s]],'k')
        plt.plot(p_new[s_idx_new[:, s]],'b--')
    '''
    '''
    # do plots
    f = plt.figure()
    s_idx,leaves = retrieve_scenarios_indexes(g)
    s_idx_new,leaves_new = retrieve_scenarios_indexes(g_new)
    values = np.array(list(nx.get_node_attributes(g, 'v').values()))
    values_new = np.array(list(nx.get_node_attributes(g_new, 'v').values()))
    cmap = plt.get_cmap('Set1')
    line_colors = cmap(np.arange(3))
    plt.plot(samples[:,:,0],alpha = 0.1)
    for s in np.arange(s_idx.shape[1]):
        plt.plot(values[s_idx[:,s]],color=line_colors[0,:],linewidth=1,alpha = 1)
        plt.plot(values_new[s_idx_new[:, s]], color=line_colors[1, :], linewidth=1, alpha=1)
    plt.plot(values[s_idx[:, s]], color=line_colors[0, :], linewidth=1, alpha=1, label='old')
    plt.plot(values_new[s_idx_new[:, s]], color=line_colors[1, :], linewidth=1, alpha=1, label='new')
    plt.legend()
    '''
    return g_new, distances