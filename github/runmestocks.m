load samplez.mat;
train_window_size = 120;
test_size = 1;
% if update only 12 months
% parfor i = (1:240/12)*12-11
%     train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:);
%     test_set = Z(Z(:,end)>= train_window_size+i & Z(:,end) <=train_window_size + i +11,:); %may
%% initialize 
error_sqr_pca = zeros(test_size,1);
error_sqr_spca = zeros(test_size,1);
error_sqr_lasso = zeros(test_size,1);
error_sqr_pls = zeros(test_size,1);
error_sqr_boost = zeros(test_size,1);
error_sqr_bag = zeros(test_size,1);
error_sqr_nn = zeros(test_size,1);


%%
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
    %% pca
    [~,Fhat_pca_all,~]=pca([x_train;x_test]); 
    k_list = [10,20];
    k = pcak(k_list,train_set(:,[1:end-2,end]),0.6);
    Fhat_pca=Fhat_pca_all(1:end-size(x_test,1),1:k);
    F_pca_new=Fhat_pca_all(end-size(x_test,1)+1:end,1:k);
    %Fit F and Y to get beta
    predictor_pca=[ones(rows(Fhat_pca),1),Fhat_pca];
    betahat_pca=(predictor_pca'*predictor_pca)\predictor_pca'* y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_pca = [ones(size(F_pca_new,1),1),F_pca_new]*betahat_pca;
    % how to save and index the errors? no need 
    % Just check dibold mariano test
    error_sqr_pca (i,1) = sum((y_test-y_hat_pca).^2)/size(y_hat_pca,1);    
    % top 1000 by market value (mve)in the testing period.
    value = e^mve;
    value_weight = e^(x_test.mve) ./ sum(e^(x_test.mve));
    table = [y_hat_pca,value]
    y_hat_pca_top = sort(table.y_hat_pca, table.value)(end-999:end);
    error_sqr_pca_top (i,1) = sum((y_test-y_hat_pca_top).^2)/size(y_hat_pca_top,1); 
    y_hat_pca_bottom = sort(table.y_hat_pca, table.value)(1:1000);
    error_sqr_pca_bottom(i,1) = sum((y_test-y_hat_pca_bottom).^2)/size(y_hat_pca_bottom,1);    

    

     
    %% spca
    
%     optw = 0.1;
%     optk = 10;
    k_list = [10,20];
    [optw,optk]=selectk(k_list,train_set(:,[1:end-2,end]),0.6);
    [Fhat_spca,A,B]=factorfit(x_train,y_train,optk,optw); %(run x(2:301-5) and y (2+5:301) to get F and beta
    % [~,sdX,meanX]=standard(x_train);
    % Fnew=(x_test-meanX(1,:))./sdX(1,:)*A;     
    F_spca_new = x_test*A;
    predictor_spca=[ones(rows(Fhat_spca),1),Fhat_spca];
    betahat_spca=(predictor_spca'*predictor_spca)\predictor_spca'*y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_spca = [ones(size(F_spca_new,1),1),F_spca_new]*betahat_spca;
    error_sqr_spca (i,1) = sum((y_test-y_hat_spca).^2)/size(y_hat_spca,1);   
    %% LASSO
    
   [Coeff_Lasso, Info_Lasso]=lasso(x_train,y_train,'Alpha',0.5,'CV',10,'Options',statset('UseParallel',false));
    %%%%if length(find(Coeff_Lasso(:,Info_Lasso.IndexMinMSE)~=0))>=round(cols(x_train)*0.01)
       Indexlasso=Info_Lasso.IndexMinMSE;
%     else
%        % keep at least 1% of the features
%        sort_mse=sort(Info_Lasso.MSE);
%        z=2;
%        while length(find(Coeff_Lasso(:,(Info_Lasso.MSE==sort_mse(z)))~=0))<round(cols(x_train)*0.01)
%            z=z+1;
%        end
%       Indexlasso=find(Info_Lasso.MSE==sort_mse(z));
%     end
    
    betahat_lasso=Coeff_Lasso(:,Indexlasso);         
    x_lasso=x_train(:,(abs(betahat_lasso)>0));
    y_hat_lasso = x_test*betahat_lasso + Info_Lasso.Intercept(Indexlasso); 
    error_sqr_lasso(i,1) = sum((y_test-y_hat_lasso).^2)/size(y_hat_lasso,1);  
    
    %% pls
    n_list = [10,4,6];
    ncomp = pls_ncomp(n_list,train_set(:,[1:end-2,end]),0.6);
    [~,~,~,~,~,~,MSE] = plsregress(x_train,y_train,ncomp);
    [~,npls]=min(MSE(2,:));
    [~,~,~,~,betapls,~,~] = plsregress(x_train,y_train,max(3,npls-1));
    y_hat_pls = [ones(size(x_test,1),1),x_test]*betapls;
    error_sqr_pls(i,1) = sum((y_test-y_hat_pls).^2)/size(y_hat_pls,1);  
    
    %% Boost
    md_boost = fitrensemble(x_train,y_train,'Method','LSBoost','NumLearningCycles',100);
    y_hat_boost = predict(md_boost,x_test);
    error_sqr_boost(i,1) = sum((y_test-y_hat_boost).^2)/size(y_hat_boost,1);  
    %% Bag
    md_bag = fitrensemble(x_train,y_train,'Method','Bag','NumLearningCycles',100);
    y_hat_bag = predict(md_bag,x_test);
    error_sqr_bag(i,1) = sum((y_test-y_hat_bag).^2)/size(y_hat_bag,1);      

    %% neural network test
    
    net.performFcn = 'mse';  % Mean Squared Error

    x_train = x_train';
    y_train = y_train';
    x_test = x_test';
    y_test = y_test';
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

    % Create a Fitting Network
    hiddenLayerSize = 2;
    net.trainParam.showWindow = false;

    net = fitnet(hiddenLayerSize,trainFcn);

    % Setup Division of Data for Training, Validation, Testing 
    [trainInd,valInd,testInd] = divideblock(size(train_set,2),0.6,0.4,0);
    % the first hidden layer use poslin (nelu)
    net.layers{1}.transferFcn = 'poslin';
    % Train the Network
    [net,tr] = train(net,x_train,y_train);
    %,'useParallel','yes'
    % performance only on the test set.
    y_hat_nn = net(x_test);
    error_sqr_nn (i,1) = sum((y_test-y_hat_nn).^2)/size(y_hat_nn,2);    
end;
msfe_pca=sum(error_sqr_pca)/test_size;
msfe_spca=sum(error_sqr_spca)/test_size;
msfe_boost = sum(error_sqr_boost)/test_size;
msfe_bag = sum(error_sqr_bag)/test_size;
msfe_lasso = sum(error_sqr_lasso)/test_size;
msfe_nn = sum(error_sqr_nn)/test_size;

r_sqr_pca = sum(error_sqr_pca)/sum(y_test.^2);
r_sqr_spca = sum(error_sqr_spca)/sum(y_test.^2);
r_sqr_boost = sum(error_sqr_boost)/sum(y_test.^2);
r_sqr_bag = sum(error_sqr_bag)/sum(y_test.^2);
r_sqr_lasso = sum(error_sqr_lasso)/sum(y_test.^2);
r_sqr_nn = sum(error_sqr_nn)/sum(y_test.^2);

sr_gain_pca = sqrt((sr_pca.^2 + r_sqr_pca)/(1-r_sqr_pca))-sr_pca;
sr_gain_spca = sqrt((sr_spca.^2 + r_sqr_spca)/(1-r_sqr_spca))-sr_spca;
sr_gain_boost = sqrt((sr_boost.^2 + r_sqr_boost)/(1-r_sqr_boost))-sr_boost;
sr_gain_bag = sqrt((sr_bag.^2 + r_sqr_bag)/(1-r_sqr_bag))-sr_bag;
sr_gain_lasso = sqrt((sr_lasso.^2 + r_sqr_lasso)/(1-r_sqr_lasso))-sr_lasso;
sr_gain_nn = sqrt((sr_nn.^2 + r_sqr_nn)/(1-r_sqr_nn))-sr_nn;

h = 1;
[~,p_value_spca] = dmtest_modified_esqr(error_sqr_pca, error_sqr_spca, h);
[~,p_value_boost] = dmtest_modified_esqr(error_sqr_pca, error_sqr_boost, h);
[~,p_value_bag] = dmtest_modified_esqr(error_sqr_pca, error_sqr_bag, h);
[~,p_value_lasso] = dmtest_modified_esqr(error_sqr_pca, error_sqr_lasso, h);
[~,p_value_nn] = dmtest_modified_esqr(error_sqr_pca, error_sqr_nn, h);


output=[msfe_pca,msfe_spca,msfe_lasso,msfe_nn,msfe_boost,msfe_bag]
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('stocksforecast','.mat');
matfile = fullfile(FolderDestination, filename);
%save(matfile);

