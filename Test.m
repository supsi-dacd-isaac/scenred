% Test the generation of the scenario tree
clear
clc
close all
addpath('../Data_preprocess/')

%% Load scenarios of bivariate data (voltage and temperature at a given point of the distribution grid)
load('data.mat');

% Plot the original 791 scenarios
figure; 
subplot(2,1,1);
plot(data{1});  title('Voltage profiles'); xlabel('time [h]'); ylabel('[V]')
subplot(2,1,2);
plot(data{2}); title('Temperature profiles');xlabel('time [h]');ylabel('[°C]')

%% Generate scenario tree 
% specify accuracy
[S_tol,P_tol,J_tol] = scenred(data, 'cityblock','tol',0.1); 
do_plots(S_tol,P_tol);

% specify number of nodes at each timestep
T = size(data{1},1);    % timesteps
N = 30;                 % number of scenarios at t==T   
[S,P,J] = scenred(data, 'cityblock','nodes',round(linspace(1,N,T))); 
do_plots(S,P);

