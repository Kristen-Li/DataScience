%Z = Z(300000:end,:);
train_window_size = 200;
test_size = 1;

%% initialize 
error_sqr_lasso = zeros(test_size,1);
error_sqr_lasso_top = zeros(test_size,1);
error_sqr_lasso_bottom = zeros(test_size,1);

rp_vw_lasso = zeros(test_size,11); 
rp_real_vw_lasso = zeros(test_size,11);
rp_ew_lasso = zeros(test_size,11); 
rp_real_ew_lasso = zeros(test_size,11);

fvu_lasso = zeros(test_size,1);
fvu_lasso_top = zeros(test_size,1);
fvu_lasso_bottom = zeros(test_size,1);

ncomp_lasso = zeros(test_size,1);
%%
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
     %% LASSO
    
%    [Coeff_Lasso, Info_Lasso]=lasso(x_train,y_train,'Standardize',false,'Alpha',0.5,'CV',2,'Options',statset('UseParallel',true));
%     %%%%if length(find(Coeff_Lasso(:,Info_Lasso.IndexMinMSE)~=0))>=round(cols(x_train)*0.01)
%        Indexlasso=Info_Lasso.IndexMinMSE;
% %     else
% %        % keep at least 1% of the features
% %        sort_mse=sort(Info_Lasso.MSE);
% %        z=2;
% %        while length(find(Coeff_Lasso(:,(Info_Lasso.MSE==sort_mse(z)))~=0))<round(cols(x_train)*0.01)
% %            z=z+1;
% %        end
% %       Indexlasso=find(Info_Lasso.MSE==sort_mse(z));
% %     end
%     
%     betahat_lasso=Coeff_Lasso(:,Indexlasso);         
%     x_lasso=x_train(:,(abs(betahat_lasso)>0));
%     y_hat_lasso = x_test*betahat_lasso + Info_Lasso.Intercept(Indexlasso); 
%     error_sqr_lasso(i,1) = sum((y_test-y_hat_lasso).^2)/size(y_hat_lasso,1);  
    
    %% NEW LASSO
    idx = round(size(x_train,1)*8/10);
    [LassoBetaEstimates,FitInfo] = lasso(x_train(1:idx,:),y_train(1:idx,:),'Standardize',false,'Options',statset('UseParallel',true));
    x_validation = x_train(idx+1:end,:);
    y_validation = y_train(idx+1:end,:);
    %Compute the FMSE of each model returned by lasso.

    y_validation_lasso = FitInfo.Intercept + x_validation*LassoBetaEstimates;
    fmseLasso = sqrt(mean((y_validation - y_validation_lasso).^2,1));
    %Plot the magnitude of the regression coefficients with respect to the shrinkage value.

%     hax = lassoPlot(LassoBetaEstimates,FitInfo);
%     L1Vals = hax.Children.XData;
%     yyaxis right
%     h = plot(L1Vals,fmseLasso,'LineWidth',2,'LineStyle','--');
%     legend(h,'FMSE','Location','SW');
%     ylabel('FMSE');
%     title('Frequentist Lasso')
    
    [min_fmseLasso,min_fmseLasso_index]=min(fmseLasso,[],2);
    ncomp_lasso(i) = max(3,FitInfo.DF(min_fmseLasso_index));
    
    fmsebestlasso = min(fmseLasso(FitInfo.DF == ncomp_lasso(i)));
    idx = fmseLasso == fmsebestlasso;
    bestLasso = [FitInfo.Intercept(idx); LassoBetaEstimates(:,idx)];
    y_hat_lasso = x_test*LassoBetaEstimates(:,idx) + FitInfo.Intercept(idx); 
    error_sqr_lasso(i,1) = sum((y_test-y_hat_lasso).^2)/size(y_hat_lasso,1);  
    
    %% vw_lasso portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,9)); % save mve before the transformation
    value_weight = value ./ sum(value);
    equal_weight = ones(size(y_test,1),1)/size(y_test,1);
    [rp_vw_lasso(i,:),rp_real_vw_lasso(i,:)]=r_portfolio_weighted(y_test,y_hat_lasso,value_weight);
    [rp_ew_lasso(i,:),rp_real_ew_lasso(i,:)]=r_portfolio_weighted(y_test,y_hat_lasso,equal_weight);

    
    %% large and small stocks by market-equity (value)
    table = [y_hat_lasso,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_lasso_top(i,1) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/1000; 
    error_sqr_lasso_bottom(i,1) = sum((table(1:1000,2)-table(1:1000,1)).^2)/1000;  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_lasso(i,:) = sum((y_test-y_hat_lasso).^2)/sum(y_test.^2);
    fvu_lasso_top(i,:) = sum((table(end-999:end,2)-table(end-999:end,1)).^2)/sum(y_test.^2);
    fvu_lasso_bottom(i,:) = sum((table(1:1000,2)-table(1:1000,1)).^2)/sum(y_test.^2);
     

end;

%% results

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_lasso = (1- sum(fvu_lasso)) *100;
r_sqr_lasso_top = (1- sum(fvu_lasso_top))*100;
r_sqr_lasso_bottom = (1- sum(fvu_lasso_bottom))*100;

%comparison of monthly out-of-sample prediction using Diebold-Mariano tests
%([~,p_value_slasso] = dmtest_modified_esqr(error_sqr_lasso, error_sqr_slasso, h);


%variable importance

%machine learning portfolios: pred, avg, ad, sr for each quantile, real 
% is right or wrong? I want the realized, not real. 
pred_vw_lasso = mean(rp_vw_lasso,1)'*100;
avg_vw_lasso = mean(rp_real_vw_lasso,1)'*100;
sd_vw_lasso = std(rp_real_vw_lasso,1)';
sr_vw_lasso = avg_vw_lasso./sd_vw_lasso/100;

pred_ew_lasso = mean(rp_ew_lasso,1)'*100;
avg_ew_lasso = mean(rp_real_ew_lasso,1)'*100;
sd_ew_lasso = std(rp_real_ew_lasso,1)';
sr_ew_lasso = avg_ew_lasso./sd_ew_lasso/100;

%%drawdowns, trunover, risk-adjusted performance of machine learning
%%portfolios (needs to save the returns of the portfolio each month)

% cumulative returns for value and equal weighted portfolio
cumul_ret_lasso = zeros(test_size,6);
cumul_ret_lasso (1,1:3) = log(1+rp_real_vw_lasso(1,[1,end-1,end]));
cumul_ret_lasso (1,4:6) = log(1+rp_real_ew_lasso(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_lasso (t+1,1:3) = sum(log(1+rp_real_vw_lasso(t+1,[1,end-1,end]))) + cumul_ret_lasso(t,1:3); 
   cumul_ret_lasso (t+1,4:6) = sum(log(1+rp_real_ew_lasso(t+1,[1,end-1,end]))) + cumul_ret_lasso(t,4:6); 
end

cumul_sorted = sort(cumul_ret_lasso);
maxdd_lasso = cumul_sorted(end,:) - cumul_sorted(1,:);
%% save results
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('model_lasso','.mat');
matfile = fullfile(FolderDestination, filename);
%save(matfile);
save('samplez','stock_level');
