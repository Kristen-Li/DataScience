clearvars -except Z;
train_window_size = 100;
test_size = 1;

%% initialize 
error_sqr_pls = zeros(test_size,1);
error_sqr_pls_top = zeros(test_size,1);
error_sqr_pls_bottom = zeros(test_size,1);

rp_vw_pls = zeros(test_size,11); 
rp_real_vw_pls = zeros(test_size,11);
rp_ew_pls = zeros(test_size,11); 
rp_real_ew_pls = zeros(test_size,11);

fvu_pls = zeros(test_size,1);
fvu_pls_top = zeros(test_size,1);
fvu_pls_bottom = zeros(test_size,1);

ncomp_pls = zeros(test_size,1);
%%
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
   %% pls
    n_list = [0,1,2,3,4,5,6,7,8,9,10];
    ncomp_pls(i) = pls_ncomp(n_list,train_set(:,[1:end-2,end]),0.7);
    [~,~,~,~,~,~,MSE] = plsregress(x_train,y_train,ncomp_pls(i));
    [~,npls]=min(MSE(2,:));
    [~,~,~,~,betapls,~,~] = plsregress(x_train,y_train,max(3,npls-1));
    y_hat_pls = [ones(size(x_test,1),1),x_test]*betapls;
    error_sqr_pls(i,1) = sum((y_test-y_hat_pls).^2)/size(y_hat_pls,1);  
    
    %% vw_pls portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,9)); % save mve before the transformation
    value_weight = value ./ sum(value);
    equal_weight = ones(size(y_test,1),1)/size(y_test,1);
    [rp_vw_pls(i,:),rp_real_vw_pls(i,:)]=r_portfolio_weighted(y_test,y_hat_pls,value_weight);
    [rp_ew_pls(i,:),rp_real_ew_pls(i,:)]=r_portfolio_weighted(y_test,y_hat_pls,equal_weight);

    
    %% large and small stocks by market-equity (value)
    table = [y_hat_pls,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_pls_top(i,1) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/1000; 
    error_sqr_pls_bottom(i,1) = sum((table(1:1000,2)-table(1:1000,1)).^2)/1000;  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_pls(i,:) = sum((y_test-y_hat_pls).^2)/sum(y_test.^2);
    fvu_pls_top(i,:) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/sum(y_test.^2);
    fvu_pls_bottom(i,:) = sum((table(1:1000,2)-table(1:1000,1)).^2)/sum(y_test.^2);
     

end;

%% results

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_pls = (1- sum(fvu_pls)) *100;
r_sqr_pls_top = (1- sum(fvu_pls_top))*100;
r_sqr_pls_bottom = (1- sum(fvu_pls_bottom))*100;

%comparison of monthly out-of-sample prediction using Diebold-Mariano tests
%([~,p_value_spls] = dmtest_modified_esqr(error_sqr_pls, error_sqr_spls, h);


%variable importance

%machine learning portfolios: pred, avg, ad, sr for each quantile, real 
% is right or wrong? I want the realized, not real. 
pred_vw_pls = mean(rp_vw_pls,1)'*100;
avg_vw_pls = mean(rp_real_vw_pls,1)'*100;
sd_vw_pls = std(rp_real_vw_pls,1)';
sr_vw_pls = avg_vw_pls./sd_vw_pls/100;

pred_ew_pls = mean(rp_ew_pls,1)'*100;
avg_ew_pls = mean(rp_real_ew_pls,1)'*100;
sd_ew_pls = std(rp_real_ew_pls,1)';
sr_ew_pls = avg_ew_pls./sd_ew_pls/100;

%%drawdowns, trunover, risk-adjusted performance of machine learning
%%portfolios (needs to save the returns of the portfolio each month)

% cumulative returns for value and equal weighted portfolio
cumul_ret_pls = zeros(test_size,6);
cumul_ret_pls (1,1:3) = log(1+rp_real_vw_pls(1,[1,end-1,end]));
cumul_ret_pls (1,4:6) = log(1+rp_real_ew_pls(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_pls (t+1,1:3) = sum(log(1+rp_real_vw_pls(t+1,[1,end-1,end]))) + cumul_ret_pls(t,1:3); 
   cumul_ret_pls (t+1,4:6) = sum(log(1+rp_real_ew_pls(t+1,[1,end-1,end]))) + cumul_ret_pls(t,4:6); 
end

cumul_sorted = sort(cumul_ret_pls);
maxdd_pls = cumul_sorted(end,:) - cumul_sorted(1,:);
%% save results
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('model_pls','.mat');
matfile = fullfile(FolderDestination, filename);
%save(matfile);