function do_plots(S,P)
%Do some plots of the scenario tree. This function works just in case of
%bivariate data. Multivariate data with n>2 is hard to visualize.

assert(length(size(S))==3,'S must be a 3-D matrix, n_timesteps x n_scenarios x n_variables')
assert(size(S,3)==2,'This function works just in case of bivariate data. Multivariate data with n>2 is hard to visualize')

T = size(P,1);
%% Probability plots
% figure; 
% imagesc(P); 
% title('Probability'); xlabel('scenario'); ylabel('time');

figure; plot(P);title('Probability')
ylabel('scenario'); xlabel('time');
xlim([1,T])
%% Scenario plots
h = figure;
cm = parula(T);
for i=1:T
    selector = P(i,:)>0;
    scatter3(ones(sum(selector),1)*i,S(i,selector,1),S(i,selector,2),P(i,selector)*1000+1,'.','MarkerEdgeColor',cm(i,:)); hold on;
end
for i=1:size(S,2)
x = 1:T;
y = S(:,i,1);
z = S(:,i,2);
plot3(x,y,z,'Color',[0,0,0,0.2]);  
end
xlabel('timestep [-]')
ylabel('variable 1')
zlabel('variable 2')
xlim([1,T])

h = figure; 
subplot 211
plot(1:T,S(:,:,1),'Color',[0,0,0,0.1]);
xlabel('timestep [-]')
ylabel('variable 1')
xlim([1,T])
subplot 212
plot(1:T,S(:,:,2),'Color',[0,0,0,0.1]);
xlabel('timestep [-]')
ylabel('variable 2')
xlim([1,T])
end