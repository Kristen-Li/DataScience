
function model_spca_dist(i_list,jobid)
train_window_size = 240;
test_size =12;
load samplez.mat;
load ncomp_spca.mat;

% if update only 12 months
% parfor i = (1:240/12)*12-11
%     train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:);
%     test_set = Z(Z(:,end)>= train_window_size+i & Z(:,end) <=train_window_size + i +11,:); %may
%% initialize 

error_sqr_spca = zeros(test_size,1);
error_sqr_spca_top = zeros(test_size,1);
error_sqr_spca_bottom = zeros(test_size,1);

rp_vw_spca = zeros(test_size,11); 
rp_real_vw_spca = zeros(test_size,11);
rp_ew_spca = zeros(test_size,11); 
rp_real_ew_spca = zeros(test_size,11);

std_vw_spca = zeros(test_size,11); 
std_real_vw_spca = zeros(test_size,11);
std_ew_spca = zeros(test_size,11); 
std_real_ew_spca = zeros(test_size,11);


fvu_spca = zeros(test_size,1);
fvu_spca_top = zeros(test_size,1);
fvu_spca_bottom = zeros(test_size,1);

k_spca = zeros(test_size,1);
w_spca = zeros(test_size,1);
y_spca_all = [];
equal_weight1_all = [];
value_weight1_all = [];
equal_weight10_all = [];
value_weight10_all = [];

q1_all = [];
q10_all = [];
%%
for i = [min(i_list):min(i_list)+11]
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
%      if i <229
%          i
%          k_spca(i,1) = ncomp_spca((ncomp_spca(:,1)==fix(i/12)*12+1),2);
%          w_spca(i,1) = ncomp_spca((ncomp_spca(:,1)==fix(i/12)*12+1),3);
%      elseif i >=229
%          k_spca(i,1) = ncomp_spca(end,2);
%          w_spca(i,1) = ncomp_spca(end,3);
%      end
%     optk = k_spca(i,1);
%     optw = w_spca(i,1);
    optw = 0.1;
    optk = 30;
      
    [Fhat_spca,A,B]=factorfit(x_train,y_train,optk,optw); 
    % F_spca_new=(x_test-meanX(1,:))./sdX(1,:)*A;     
    F_spca_new = x_test*A;
    predictor_spca=[ones(rows(Fhat_spca),1),Fhat_spca]; 
    betahat_spca=(predictor_spca'*predictor_spca)\predictor_spca'*y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_spca = [ones(size(F_spca_new,1),1),F_spca_new]*betahat_spca;
    y_spca_all = [y_spca_all;y_test];
    error_sqr_spca (i,1) = sum((y_test-y_hat_spca).^2)/size(y_hat_spca,1);   
    
    %% vw_spca portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,48)); % save mve before the transformation
    equal = ones(size(x_test,1));
    [rp_vw_spca(i,:),rp_real_vw_spca(i,:),std_vw_spca(i,:),std_real_vw_spca(i,:),value_weight1,value_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_spca,value);
    [rp_ew_spca(i,:),rp_real_ew_spca(i,:),std_ew_spca(i,:),std_real_ew_spca(i,:),equal_weight1,equal_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_spca,equal);
    value_weight1_all = [value_weight1_all;value_weight1];
    equal_weight1_all = [equal_weight1_all;equal_weight1];
    value_weight10_all = [value_weight10_all;value_weight10];
    equal_weight10_all = [equal_weight10_all;equal_weight10];

    q1_all = [q1_all;q1]; %q1 is logical values of size(x_test,1),the stocks in first quantile
    q10_all = [q10_all;q10];


    
    %% large and small stocks by market-equity (value)
    if i == 240
        continue
    end 

    table = [y_hat_spca,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_spca_top(i,1) = sum((table(end-499:end,2)-table(end-499:end,1)).^2); 
    error_sqr_spca_bottom(i,1) = sum((table(1:500,2)-table(1:500,1)).^2);  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_spca(i,:) = sum(y_test.^2);
    fvu_spca_top(i,:) = sum(table(end-499:end,2).^2);
    fvu_spca_bottom(i,:) = sum(table(1:500,2).^2);
     

end;

%% results

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_spca = (1- sum(error_sqr_spca)/sum(fvu_spca)) *100;
r_sqr_spca_top = (1- sum(error_sqr_spca)/sum(fvu_spca_top))*100;
r_sqr_spca_bottom = (1- sum(error_sqr_spca)/sum(fvu_spca_bottom))*100;

%comparison of monthly out-of-sample prediction using Diebold-Mariano tests
%([~,p_value_sspca] = dmtest_modified_esqr(error_sqr_spca, error_sqr_sspca, h);


%variable importance

%machine learning portfolios: pred, avg, ad, sr for each quantile, real 
% is right or wrong? I want the realized, not real. 
pred_vw_spca = mean(rp_vw_spca,1)'*100;
avg_vw_spca = mean(rp_real_vw_spca,1)'*100;
sd_vw_spca = std(rp_real_vw_spca,1)';
sr_vw_spca = avg_vw_spca./sd_vw_spca/100;

pred_ew_spca = mean(rp_ew_spca,1)'*100;
avg_ew_spca = mean(rp_real_ew_spca,1)'*100
sd_ew_spca = std(rp_real_ew_spca,1)';
sr_ew_spca = avg_ew_spca./sd_ew_spca/100;


%%drawdowns, trunover, risk-adjusted performance of machine learning
%%portfolios (needs to save the returns of the portfolio each month)

% cumulative returns for value and equal weighted portfolio, L, H , H-L
cumul_ret_spca = zeros(test_size,6);
cumul_ret_spca (1,1:3) = log(1+rp_real_vw_spca(1,[1,end-1,end]));
cumul_ret_spca (1,4:6) = log(1+rp_real_ew_spca(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_spca (t+1,1:3) = sum(log(1+rp_real_vw_spca(t+1,[1,end-1,end]))) + cumul_ret_spca(t,1:3); 
   cumul_ret_spca (t+1,4:6) = sum(log(1+rp_real_ew_spca(t+1,[1,end-1,end]))) + cumul_ret_spca(t,4:6); 
end

cumul_sorted = sort(cumul_ret_spca);
maxdd_spca = cumul_sorted(end,:) - cumul_sorted(1,:);

%% save results
clear Z train_set x_train y_train x_test y_test Fhat_spca Fhat_spca_all predictor_spca;
FolderDestination='stocks_output';
filename=strcat('model_spca_',num2str(i),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');
