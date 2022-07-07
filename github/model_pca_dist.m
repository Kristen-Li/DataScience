function model_pca_dist(i_list,jobid)

train_window_size = 240;
test_size =12;                                                                         

Z= readtable('ranked500.csv','PreserveVariableNames',true);
Z= removevars(Z,{'Var1'}); 
Z= rmmissing(Z);
Z=sortrows(Z,{'DATE','permno'},{'ascend','ascend'});

date = [];
time_index = [];
permno = [];
eret_test = [];
eret_pred = [] ;
rank_mvel1 = [];

colnames =  Z.Properties.VariableNames';
mve_index = find(contains(colnames, 'rank_mvel1'));
xcols = setdiff(colnames,{ 'date', 'permno','Var1','DATE', 'gvkey', 'sic','sic2','ffi49','eret','ret','fret','time_index','dp','ep_mac','tbl',	'tms',	'dfy',	'svar'});
macols = {'dp','ep_mac','tbl',	'tms',	'dfy',	'svar'};
firmcols = setdiff(xcols,macols);

for i = [i_list:i_list+test_size-1]

    train_set = Z((Z.time_index >=i )& (Z.time_index <= train_window_size -1 + i),:); % subsample 100 features (Z.time_index >=i )& 
    test_set = Z(Z.time_index == train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test    
    x_train = train_set{:,xcols};
    y_train = train_set{:,{'eret'}};
    x_test = test_set{:,xcols};
    y_test = test_set{:,{'eret'}};        
 
    %% pca
    [~,Fhat_pca,~]=pca([x_train],'Centered',false); 
    k_list = [5,20,30];
    %k = pcak(k_list,train_set(:,[1:end-2,end]),0.2);

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

    date = [date;test_set.date];
    time_index = [time_index;test_set.time_index];
    permno = [permno;test_set.permno];
    rank_mvel1 = [rank_mvel1;test_set.rank_mvel1];
    eret_test = [eret_test;y_test];
    eret_pred = [eret_pred;y_hat_pca];
    nfactors = optk;
end

result = table(time_index,date,permno,eret_test,eret_pred, rank_mvel1);
clearvars -except result nfactors i_list;
FolderDestination='stocks_output';
filename=strcat('model_pca_',num2str(i_list),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');
