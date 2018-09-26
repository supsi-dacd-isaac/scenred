import numpy as np
from scipy.spatial.distance import pdist,squareform

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
    for key in ('nodes','tol'):
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
            z[z == 0] = infty  # set prob. of removed scenario to inf in order to ignore them
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
            delta_p = delta_p + dp_min
            delta_rel = delta_p / delta_max

        if i > 0:
            # Update available scenarios in the previous timestep
            J[i - 1,:] = J[i,:]

            # update previous timestep probabilities
            P[i - 1,:] = P[i,:]
            D[~J[i,:], ~J[i,:]] = infty

        print('Branches t=%i: %i' %  (i, branches))

    S = S[:, J[-1,:] > 0,:]
    P = P[:, J[-1,:] > 0]

    # obtain the match matrix
    Me = np.zeros((S.shape[1],S.shape[1],T))
    for i in np.arange(T):
        D = get_dist(S[i, :, 0].reshape(-1, 1), pars['metric'])
        #D[np.eye(D.shape[0],dtype=bool)] = 1
        Me[:,:,i] = D==0

    Me = np.swapaxes(Me,2,0)

    return S, P, J, Me
