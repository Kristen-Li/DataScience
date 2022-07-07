function model_pca_dist(i_list,jobid)
% i_list = 1;
% jobid = 6;
train_window_size = 240;
test_size =12;                                                                         

Z= readtable('rankednomissing.csv','PreserveVariableNames',true);
Z= removevars(Z,{'Var1'}); 
Z= rmmissing(Z);
Z=sortrows(Z,{'DATE','permno'},{'ascend','ascend'});

num  = zeros(test_size,1);

error_sqr_pca = zeros(test_size,1);
error_sqr_pca_top = zeros(test_size,1);
error_sqr_pca_bottom = zeros(test_size,1);

rp_vw_pca = zeros(test_size,11); 
rp_real_vw_pca = zeros(test_size,11);
rp_ew_pca = zeros(test_size,11); 
rp_real_ew_pca = zeros(test_size,11);

std_vw_pca = zeros(test_size,11); 
std_real_vw_pca = zeros(test_size,11);
std_ew_pca = zeros(test_size,11); 
std_real_ew_pca = zeros(test_size,11);


fvu_pca = zeros(test_size,1);
fvu_pca_top = zeros(test_size,1);
fvu_pca_bottom = zeros(test_size,1);

k_pca = zeros(test_size,1);
y_pca_all = [];
equal_weight1_all = [];
value_weight1_all = [];
equal_weight10_all = [];
value_weight10_all = [];
q1_all = [];
q10_all = [];
colnames =  Z.Properties.VariableNames';
mve_index = find(contains(colnames, 'rank_mvel1'));
xcols = setdiff(colnames,{ 'date', 'permno','Var1','DATE', 'gvkey', 'sic','sic2','ffi49','eret','ret','fret','time_index','dp','ep_mac','tbl',	'tms',	'dfy',	'svar'});
macols = {'dp','ep_mac','tbl',	'tms',	'dfy',	'svar'};
firmcols = setdiff(xcols,macols);

for i = [min(i_list):min(i_list)+11]
    train_set = Z((Z.time_index >=i )& (Z.time_index <= train_window_size -1 + i),:); % subsample 100 features (Z.time_index >=i )& 
    test_set = Z(Z.time_index == train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test    
    x_train = train_set{:,xcols};
    y_train = train_set{:,{'eret'}};
    x_test = test_set{:,xcols};
    y_test = test_set{:,{'eret'}};
    num(i,1) = size(x_test,1);

    %% pca
    [~,Fhat_pca,~]=pca([x_train],'Centered',false); 
%     k_list = [1:20];
%     k = pcak(k_list,train_set(:,[1:end-2,end]),0.2);

    % CHANGED HERE k = k_pca(i,1);
    optk = 30;     
    Fhat_pca = Fhat_pca(:,1:optk);
    A=pinv(x_train)*Fhat_pca;
    F_pca_new = x_test*A;    
    %Fit F and Y to get beta
    predictor_pca=[ones(rows(Fhat_pca),1),Fhat_pca];
    betahat_pca=pinv(predictor_pca)* y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_pca = [ones(size(F_pca_new,1),1),F_pca_new]*betahat_pca;

end

%% save results
clear Z train_set x_train y_train x_test Fhat_pca Fhat_pca_all predictor_pca;
FolderDestination='stocks_output';
filename=strcat('model_pca_',num2str(i),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');
