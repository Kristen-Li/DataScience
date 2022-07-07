%Z = Z(1:300000,:);
train_window_size = 3;
test_size = 2;
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
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,[1:100, end-1:end]); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,[1:100, end-1:end]); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
    %% pca
    [~,Fhat_pca_all,~]=pca([x_train;x_test]); 
    k_list = [1,5,10,20,30,40];
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
     
    %% spca
    
%     optw = 0.1;
%     optk = 10;
    k_list = [1,5];
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

output=[msfe_pca,msfe_spca,msfe_lasso,msfe_nn,msfe_boost,msfe_bag];
mkdir('stocks_output')
FolderDestination='stocks_output';
filename=strcat('j',int2str(jobid),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile);

