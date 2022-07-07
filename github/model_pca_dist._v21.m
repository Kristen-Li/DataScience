
function model_pca_dist(i_list,jobid)
train_window_size = 240;
test_size =12;
load sample.mat;
Z = Z(:,73:end); %exclude sic dummies CHANGE HERE
load ncomp_pca.mat;
num  = 0;
% 
% if update only 12 months
% parfor i = (1:240/12)*12-11
%     train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:);
%     test_set = Z(Z(:,end)>= train_window_size+i & Z(:,end) <=train_window_size + i +11,:); %may
%% initialize 

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


% fvu_pca = zeros(test_size,1);
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
R_sqr = 0;
%%
for i = [min(i_list):min(i_list)+11]
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    %CHANGE HERE
    %x_train = standard(train_set(:,1:end-2));
    %y_train = standard(train_set(:,end-1));
    %x_test = standard(test_set(:,1:end-2));
    %y_test = standard(test_set(:,end-1));
    x_train = standard(train_set(:,1:end-2));
    y_train = (train_set(:,end-1));
    x_test = standard(test_set(:,1:end-2));
    y_test = (test_set(:,end-1));

    %% pca
    [~,Fhat_pca,~]=pca([x_train],'Centered',false); 
%     k_list = [1:20];
%     k = pcak(k_list,train_set(:,[1:end-2,end]),0.2);
    if i <229
        k_pca(i,1) = ncomp_pca((ncomp_pca(:,1)==fix(i/12)*12+1),2);
    elseif i >=229
        k_pca(i,1) = ncomp_pca(end,2);
    end
    % CHANGED HERE k = k_pca(i,1);
    k = 6;
   % Fhat_pca=Fhat_pca_all(1:end-size(x_test,1),1:k);
   % F_pca_new=Fhat_pca_all(end-size(x_test,1)+1:end,1:k);
    
    Fhat_pca = Fhat_pca(:,1:k);
    A=pinv(x_train'*x_train)*x_train'*Fhat_pca;

    F_pca_new = x_test*A;

    
    %Fit F and Y to get beta
    predictor_pca=[ones(rows(Fhat_pca),1),Fhat_pca];
    betahat_pca=(predictor_pca'*predictor_pca)\predictor_pca'* y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_pca = [ones(size(F_pca_new,1),1),F_pca_new]*betahat_pca;
    y_pca_all = [y_pca_all;y_test];    % save error squared (mean among samples at time i)

    error_sqr_pca (i,1) = sum((y_test-y_hat_pca).^2);    
    
    %% vw_pca portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,48)); % save mve before the transformation
    equal = ones(size(x_test,1));
    [rp_vw_pca(i,:),rp_real_vw_pca(i,:),std_vw_pca(i,:),std_real_vw_pca(i,:),value_weight1,value_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_pca,value);
    [rp_ew_pca(i,:),rp_real_ew_pca(i,:),std_ew_pca(i,:),std_real_ew_pca(i,:),equal_weight1,equal_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_pca,equal);
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

    table = [y_hat_pca,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_pca_top(i,1) = sum((table(end-499:end,2)-table(end-499:end,1)).^2); % sum of squared residuals
    error_sqr_pca_bottom(i,1) = sum((table(1:500,2)-table(1:500,1)).^2);  
    fvu_pca(i,:) = sum((y_test).^2); % total sum of squares
    fvu_pca_top(i,:) = sum(table(end-499:end,2).^2);
    fvu_pca_bottom(i,:) = sum(table(1:500,2).^2);
    R_sqr_temp = 1-sum(y_hat_pca-y_test).^2/sum(y_test).^2
    R_sqr = R_sqr + R_sqr_temp
end

%% save results
clear Z train_set x_train y_train x_test Fhat_pca Fhat_pca_all predictor_pca;
FolderDestination='stocks_output';
filename=strcat('model_pca_',num2str(i),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');
