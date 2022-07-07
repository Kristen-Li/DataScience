function tune_pca(i_list,jobid)
load samplez.mat;
load delete.mat;
train_window_size = 240;
k_pca = [];
%%
for i = i_list
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    k_list = [5,10,15,20,25,30,35,40,45,50,53,56,60,65,70,75,80,85,90];
    k = pcak(k_list,train_set(:,[1:end-2,end]),0.7);
    k_pca = [k_pca;k];
end
k_tune = [i_list',k_pca]
%% results
filename = strcat('k_pca_',int2str(jobid),'.mat');
save(filename,'k_tune')

