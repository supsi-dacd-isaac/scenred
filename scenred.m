function [S,P,J,Me] = scenred(samples, metric,varargin)
%Build a scenario tree reducing N observed samplings from a multivariate
%distribution. The algorithm is described in "Nicole Growe-Kuska,
% Holger Heitsch and Werner Romisch - Scenario Reduction Scenario Tree 
% Construction for Power Management Problems".
% Inputs: 
%         samples: n cell array, where n is the number of signals observed,
%         each cell of dimension T*n_obs, where T is the number of observed
%         timesteps and n_obs is the original number of observed scenarios. 
%         The scenarios in the different cells are supposed to be 
%         statistically dependent (sampled from historical data or from 
%         a copula)
%         metric: in {'euclidean','cityblock'}, used for 
%                 Kantorovich-Rubinstein metric
%         varargin: {'nodes','tol'} methods for building the tree
%                   -nodes vector with length T, increasingly monotone,
%                   with the specified number of nodes per timestep
%                   -tol: tolerance on Dist/Dist_0, where Dist is the 
%                         distance after aggregating scenarios, and Dist_0
%                         is the distance between all the scenarios and a
%                         tree with only one scenario. In any case, at
%                         least one scenario is selected.
% Outputs:
%          S: T*n_obs*n matrix, containing all the path in the scenario
%          tree
%          P: T*n_obs matrix, containing evolviong probabilities for each
%          scenario
%          J: T*n_obs matrix, indicating non-zero entries of P (if a
%          scenario is present at a given timestep)
%          Me: T*n_scen*n_scen matrix, containing which scenario is linked 
%          with which at time t

T = size(samples{1},1);
n_obs = size(samples{1},2);
defaultNodes = ones(size(samples{1},1),1);
defaultTol = 10;
expectedStrings = {'euclidean','cityblock'};

% parse input
p = inputParser;
ismonotone = @(x) all(diff(x))>=0; % number of nodes must be increasingly monotone
validVect= @(x) isnumeric(x) && isvector(x) && length(x)==T && all(diff(x)>=0) && ismonotone(x);
validScal= @(x) isnumeric(x) && isscalar(x);
validString = @(x) any(validatestring(x,expectedStrings));
addRequired(p,'metric',validString);
addOptional(p,'nodes', defaultNodes, validVect);
addOptional(p,'tol', defaultTol, validScal);

parse(p,metric,varargin{:});
tol = p.Results.tol;
nodes = p.Results.nodes;

% Obtain the observation matrix, size (n*T) x n_obs, from which to compute distances. Normalize observations 
X = [];
S = [];
for i=1:length(samples)
    V = samples{i};
    V_norm = (V-repmat(mean(V,2),1,size(V,2)))./(repmat(std(V,[],2)+1e-6,1,size(V,2)));
    X = [X;V_norm];
    S(:,:,i) = V;
end
X = X';

% Get distance between scenarios
D = get_dist(X,metric);
D = D + eye(size(D))*(1 + max(D(:))); % ignore self-distance (D diagonal)

% generate the tolerance vector
if all(nodes==defaultNodes)
    Tol = fliplr(tol./(1.5.^(T-[1:T]+1)));
    Tol(1) = inf;
else
    Tol = inf(1,T);
end
% ;
% initialize selected scenarios at each timestep
J = true(T,n_obs);
% initialize linking matrix, to keep track of which scenario has merged at
% which timestep
L = zeros(n_obs,n_obs); 
% initialize scenario probability at each timestep
P = ones(T,n_obs)/n_obs;
% Initialize the number of branches in the last stage
branches = n_obs; 
% for all the timesteps
for i=fliplr(1:T) % aggregate scenarios using a backward strategy
    
    delta_rel = 0;
    delta_p = 0;
    D_i = D;
    
    % Compute the minimum cost of approximating the current scenarios, up
    % to time i, with just one scenario
    sel_idx = repmat(logical([ones(1,i),zeros(1,T-i)]),1,length(samples));
    D_j = get_dist(X(J(i,:),sel_idx),metric);
    D_j(logical(eye(size(D_j)))) = 0;   % do not consider self-distance
    delta_max = min(sum(D_j));
     
    while delta_rel<Tol(i) && branches>nodes(i)
        D_i(~J(i,:),:) = inf;                         % set distance of discarded scenarios to infinity, ignoring them
        D_i(:,~J(i,:)) = inf;
        [d_s,idx_s] = sort(D_i);                      % sort distances with respect to the non-discarded scenarios
        z = d_s(1,:).*P(i,:);                         % vector of weighted probabilities
        z(z==0) = inf;                                % set prob. of removed scenario to inf in order to ignore them
        [dp_min, idx_rem] = min(z);                   % find the scenario which cause the smallest p*d-deviation when merged, and its index          
        idx_aug = idx_s(1,idx_rem);                   % retrieve who's being augmented with the probability of idx_rem
        J(i,idx_rem) = false;                         % mark it as a removed scenarion in the current timestep              
        P(i,idx_aug) = P(i,idx_rem) + P(i,idx_aug);   % add the probability of the removed scenario to the closest scenario
        P(i,idx_rem) = 0;                             % set probability of removed scenarios to 0
        branches = sum(P(i,:)>0);                     % count remaining branches 
        L(idx_aug,idx_rem) = 1;                       % keep track of which scenario has merged
        L(idx_aug,L(idx_rem,:)>0) = 1;                % the scenario who's been augmented heredit all the scenarios previously merged with the removed scenario
        S(1:i,idx_rem,:) =  S(1:i,idx_aug,:);         % make the merged scenarios equal up to the root node
        to_merge_idx = find(L(idx_rem,:));            % make all the scenario previously merged with the removed one equal to the one in idx_aug, up to the root node
        for j=1:length(to_merge_idx)
            S(1:i,to_merge_idx(j),:) =  S(1:i,idx_aug,:);
        end
        
        % update the differential accuracy
        delta_p = delta_p + dp_min;
        delta_rel = delta_p/delta_max;
    end
    
    if i>1
        % Update available scenarios in the previous timestep
        J(i-1,:) = J(i,:);

        % update previous timestep probabilities
        P(i-1,:) = P(i,:);
        D(~J(i,:),~J(i,:)) = inf;
    end
    fprintf('\nBranches t=%i: %i',[i,branches])
    
end

% Keep only scenarios which are present at the end (t==T)
S = S(:,J(end,:)>0,:);
P = P(:,J(end,:)>0,:);
for i=1:T
    Me(:,:,i) = get_dist(S(i,:,1)',metric)==0;
end
Me = permute(Me,[3,1,2]);

end

function D = get_dist(X,metric)
D = pdist2(X,X,metric);
end