%Z = Z(300000:end,:);
clearvars -except Z;
train_window_size = 10;
test_size = 1;

%% initialize 
error_sqr_boost = zeros(test_size,1);
error_sqr_boost_top = zeros(test_size,1);
error_sqr_boost_bottom = zeros(test_size,1);

rp_vw_boost = zeros(test_size,11); 
rp_real_vw_boost = zeros(test_size,11);
rp_ew_boost = zeros(test_size,11); 
rp_real_ew_boost = zeros(test_size,11);

fvu_boost = zeros(test_size,1);
fvu_boost_top = zeros(test_size,1);
fvu_boost_bottom = zeros(test_size,1);

ncomp_boost = zeros(test_size,1);
%%
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
   %% boost
    md_boost = fitrensemble(x_train,y_train,'Method','LSBoost','NumLearningCycles',100, 'Learners',templateTree('NumVariablesToSample',10,'Surrogate','on'));
    y_hat_boost = predict(md_boost,x_test);
    error_sqr_boost(i,1) = sum((y_test-y_hat_boost).^2)/size(y_hat_boost,1); 
    
    %% vw_boost portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,9)); % save mve before the transformation
    value_weight = value ./ sum(value);
    equal_weight = ones(size(y_test,1),1)/size(y_test,1);
    [rp_vw_boost(i,:),rp_real_vw_boost(i,:)]=r_portfolio_weighted(y_test,y_hat_boost,value_weight);
    [rp_ew_boost(i,:),rp_real_ew_boost(i,:)]=r_portfolio_weighted(y_test,y_hat_boost,equal_weight);

    
    %% large and small stocks by market-equity (value)
    table = [y_hat_boost,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_boost_top(i,1) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/1000; 
    error_sqr_boost_bottom(i,1) = sum((table(1:1000,2)-table(1:1000,1)).^2)/1000;  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_boost(i,:) = sum((y_test-y_hat_boost).^2)/sum(y_test.^2);
    fvu_boost_top(i,:) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/sum(y_test.^2);
    fvu_boost_bottom(i,:) = sum((table(1:1000,2)-table(1:1000,1)).^2)/sum(y_test.^2);
     

end;

%% results

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_boost = (1- sum(fvu_boost)) *100;
r_sqr_boost_top = (1- sum(fvu_boost_top))*100;
r_sqr_boost_bottom = (1- sum(fvu_boost_bottom))*100;

%comparison of monthly out-of-sample prediction using Diebold-Mariano tests
%([~,p_value_sboost] = dmtest_modified_esqr(error_sqr_boost, error_sqr_sboost, h);


%variable importance

%machine learning portfolios: pred, avg, ad, sr for each quantile, real 
% is right or wrong? I want the realized, not real. 
pred_vw_boost = mean(rp_vw_boost,1)'*100;
avg_vw_boost = mean(rp_real_vw_boost,1)'*100;
sd_vw_boost = std(rp_real_vw_boost,1)';
sr_vw_boost = avg_vw_boost./sd_vw_boost/100;

pred_ew_boost = mean(rp_ew_boost,1)'*100;
avg_ew_boost = mean(rp_real_ew_boost,1)'*100;
sd_ew_boost = std(rp_real_ew_boost,1)';
sr_ew_boost = avg_ew_boost./sd_ew_boost/100;

%%drawdowns, trunover, risk-adjusted performance of machine learning
%%portfolios (needs to save the returns of the portfolio each month)

% cumulative returns for value and equal weighted portfolio
cumul_ret_boost = zeros(test_size,6);
cumul_ret_boost (1,1:3) = log(1+rp_real_vw_boost(1,[1,end-1,end]));
cumul_ret_boost (1,4:6) = log(1+rp_real_ew_boost(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_boost (t+1,1:3) = sum(log(1+rp_real_vw_boost(t+1,[1,end-1,end]))) + cumul_ret_boost(t,1:3); 
   cumul_ret_boost (t+1,4:6) = sum(log(1+rp_real_ew_boost(t+1,[1,end-1,end]))) + cumul_ret_boost(t,4:6); 
end

cumul_sorted = sort(cumul_ret_boost);
maxdd_boost = cumul_sorted(end,:) - cumul_sorted(1,:);
%% save results
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('model_boost','.mat');
matfile = fullfile(FolderDestination, filename);
%save(matfile);