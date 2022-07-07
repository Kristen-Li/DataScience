tic
load samplez.mat
load k_tuned.mat
train_window_size = 240;
test_size = 240;
% if update only 12 months
% parfor i = (1:240/12)*12-11
%     train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:);
%     test_set = Z(Z(:,end)>= train_window_size+i & Z(:,end) <=train_window_size + i +11,:); %may
%% initialize 
y_hat_pca = zeros(test_size,1);
error_sqr_pca = zeros(test_size,1);
error_sqr_pca_top = zeros(test_size,1);
error_sqr_pca_bottom = zeros(test_size,1);

rp_vw_pca = zeros(test_size,11); 
rp_real_vw_pca = zeros(test_size,11);
rp_ew_pca = zeros(test_size,11); 
rp_real_ew_pca = zeros(test_size,11);

fvu_pca = zeros(test_size,1);
fvu_pca_top = zeros(test_size,1);
fvu_pca_bottom = zeros(test_size,1);

k_pca = zeros(test_size,1);
%%
num = 0;
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
    %% pca
    [~,Fhat_pca_all,~]=pca([x_train;x_test],'Centered',false); 
%     k_list = [1:20];
%     k = pcak(k_list,train_set(:,[1:end-2,end]),0.2);
    k_pca(i,1) = k_tuned((k_tuned(:,1)==fix(119/12)*12+1),2);
    k = k_pca(i,1);
    Fhat_pca=Fhat_pca_all(1:end-size(x_test,1),1:k);
    F_pca_new=Fhat_pca_all(end-size(x_test,1)+1:end,1:k);
    %Fit F and Y to get beta
    predictor_pca=[ones(rows(Fhat_pca),1),Fhat_pca];
    betahat_pca=(predictor_pca'*predictor_pca)\predictor_pca'* y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_pca(i+num:i+num+size(test_set,1)-1,1) = [ones(size(F_pca_new,1),1),F_pca_new]*betahat_pca;
    y_hat_pca_temp=y_hat_pca(i+num:i+num+size(test_set,1)-1,1);
    % save error squared (mean among samples at time i)
    error_sqr_pca (i,1) = sum((y_test-y_hat_pca_temp).^2)/size(y_test,1);    
    num = size(test_set,1);
    %% vw_pca portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,9)); % save mve before the transformation
    value_weight = value ./ sum(value);
    equal_weight = ones(size(y_test,1),1)/size(y_test,1);
    [rp_vw_pca(i,:),rp_real_vw_pca(i,:)]=r_portfolio_weighted(y_test,y_hat_pca_temp,value_weight);
    [rp_ew_pca(i,:),rp_real_ew_pca(i,:)]=r_portfolio_weighted(y_test,y_hat_pca_temp,equal_weight);

    
    %% large and small stocks by market-equity (value)
    table = [y_hat_pca_temp,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_pca_top(i,1) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/1000; 
    error_sqr_pca_bottom(i,1) = sum((table(1:1000,2)-table(1:1000,1)).^2)/1000;  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_pca(i,:) = sum((y_test-y_hat_pca_temp).^2)/sum(y_test.^2);
    fvu_pca_top(i,:) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/sum(y_test.^2);
    fvu_pca_bottom(i,:) = sum((table(1:1000,2)-table(1:1000,1)).^2)/sum(y_test.^2);     

end

%% results

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_pca = (1- sum(fvu_pca)) *100;
r_sqr_pca_top = (1- sum(fvu_pca_top))*100;
r_sqr_pca_bottom = (1- sum(fvu_pca_bottom))*100;
%comparison of monthly out-of-sample prediction using Diebold-Mariano tests
%([~,p_value_spca] = dmtest_modified_esqr(error_sqr_pca, error_sqr_spca, h);


%variable importance

%machine learning portfolios: pred, avg, ad, sr for each quantile, real 
% is right or wrong? I want the realized, not real. 
pred_vw_pca = mean(rp_vw_pca,1)'*100;
avg_vw_pca = mean(rp_real_vw_pca,1)'*100;
sd_vw_pca = std(rp_real_vw_pca,1)';
sr_vw_pca = avg_vw_pca./sd_vw_pca/100;

pred_ew_pca = mean(rp_ew_pca,1)'*100;
avg_ew_pca = mean(rp_real_ew_pca,1)'*100;
sd_ew_pca = std(rp_real_ew_pca,1)';
sr_ew_pca = avg_ew_pca./sd_ew_pca/100;


%%drawdowns, trunover, risk-adjusted performance of machine learning
%%portfolios (needs to save the returns of the portfolio each month)

% cumulative returns for value and equal weighted portfolio
cumul_ret_pca = zeros(test_size,6);
cumul_ret_pca (1,1:3) = log(1+rp_real_vw_pca(1,[1,end-1,end]));
cumul_ret_pca (1,4:6) = log(1+rp_real_ew_pca(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_pca (t+1,1:3) = sum(log(1+rp_real_vw_pca(t+1,[1,end-1,end]))) + cumul_ret_pca(t,1:3); 
   cumul_ret_pca (t+1,4:6) = sum(log(1+rp_real_ew_pca(t+1,[1,end-1,end]))) + cumul_ret_pca(t,4:6); 
end

cumul_sorted = sort(cumul_ret_pca);
maxdd_pca = cumul_sorted(end,:) - cumul_sorted(1,:);
%% save results
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('model_pca','.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile);
toc
