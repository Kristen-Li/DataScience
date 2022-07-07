function model_nn_dist(i_list,jobid)
train_window_size = 240;
test_size =12;
load sample.mat;
Z = Z(:,73:end); %exclude sic dummies CHANGE HERE
num  = 0;
% 
% if update only 12 months
% parfor i = (1:240/12)*12-11
%     train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:);
%     test_set = Z(Z(:,end)>= train_window_size+i & Z(:,end) <=train_window_size + i +11,:); %may
%% initialize 

error_sqr_nn = zeros(test_size,1);
error_sqr_nn_top = zeros(test_size,1);
error_sqr_nn_bottom = zeros(test_size,1);

rp_vw_nn = zeros(test_size,11); 
rp_real_vw_nn = zeros(test_size,11);
rp_ew_nn = zeros(test_size,11); 
rp_real_ew_nn = zeros(test_size,11);

std_vw_nn = zeros(test_size,11); 
std_real_vw_nn = zeros(test_size,11);
std_ew_nn = zeros(test_size,11); 
std_real_ew_nn = zeros(test_size,11);


% fvu_nn = zeros(test_size,1);
fvu_nn_top = zeros(test_size,1);
fvu_nn_bottom = zeros(test_size,1);

k_nn = zeros(test_size,1);
y_nn_all = [];
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

 
    optw = 0.5;
    optk = 20;

  
    linet = nnfit(train_set, 0.7);
    y_hat_nn = linet(x_test');
    y_hat_nn = y_hat_nn';    
    
    y_hat_nn_all = [y_hat_nn_all;y_hat_nn];
    error_sqr_nn (i,1) = sum((y_test-y_hat_nn).^2)/size(y_hat_nn,1);   
    



    
    %% vw_nn portfolio returns
    % top 1000 by market value (mve)in the testing period. the 9th column
    % is log(value)    
    value = exp(x_test(:,48)); % save mve before the transformation
    equal = ones(size(x_test,1));
    [rp_vw_nn(i,:),rp_real_vw_nn(i,:),std_vw_nn(i,:),std_real_vw_nn(i,:),value_weight1,value_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_nn,value);
    [rp_ew_nn(i,:),rp_real_ew_nn(i,:),std_ew_nn(i,:),std_real_ew_nn(i,:),equal_weight1,equal_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_nn,equal);
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

    table = [y_hat_nn,y_test,x_test(:,9)];
    table = sortrows(table,3);
    error_sqr_nn_top(i,1) = sum((table(end-499:end,2)-table(end-499:end,1)).^2); % sum of squared residuals
    error_sqr_nn_bottom(i,1) = sum((table(1:500,2)-table(1:500,1)).^2);  
    fvu_nn(i,:) = sum((y_test).^2); % total sum of squares
    fvu_nn_top(i,:) = sum(table(end-499:end,2).^2);
    fvu_nn_bottom(i,:) = sum(table(1:500,2).^2);
end

%% save results
clear Z train_set x_train y_train x_test Fhat_nn Fhat_nn_all predictor_nn;
FolderDestination='st_output';
filename=strcat('model_nn_',num2str(i),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');
