load samplez.mat
train_window_size = 120;
test_size = 1;
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

fvu_nn = zeros(test_size,1);
fvu_nn_top = zeros(test_size,1);
fvu_nn_bottom = zeros(test_size,1);

y_hat_nn_all = [];
equal_weight1_all = [];
value_weight1_all = [];
equal_weight10_all = [];
value_weight10_all = [];
q1_all = [];
q10_all = [];
%%
for i = 1:test_size
    train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    test_set = Z(Z(:,end)== train_window_size+i,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
    

    %% neural network test


    x_train = x_train';
    y_train = y_train';

    % Create a Fitting Network
    hiddenLayer1Size = 32;
    hiddenLayer2Size = 16;
    hiddenLayer3Size = 8;
    hiddenLayer4Size = 4;

    net = fitnet([hiddenLayer1Size hiddenLayer2Size hiddenLayer3Size hiddenLayer4Size]);
    net.trainFcn = 'traingdx';
    net.trainParam.epochs = 300;
    net.trainParam.lr = 0.01;
    net.performFcn = 'mse';  % Mean Squared Error



    % Setup Division of Data for Training, Validation, Testing 
    [trainInd,valInd,testInd] = divideblock(size(train_set,2),0.7,0.3,0);
    % the first hidden layer use poslin (nelu)
    net.layers{1}.transferFcn = 'poslin';
    net.layers{2}.transferFcn = 'poslin';
    net.layers{3}.transferFcn = 'poslin';
    net.layers{4}.transferFcn = 'poslin';

    % Train the Network
    [linet,tr] = train(net,x_train,y_train);%,'useParallel','yes');
    save linet
    load linet
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
    [rp_vw_nn(i,:),rp_real_vw_nn(i,:),value_weight1,value_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_nn,value);
    [rp_ew_nn(i,:),rp_real_ew_nn(i,:),equal_weight1,equal_weight10,q1,q10]=r_portfolio_weighted(y_test,y_hat_nn,equal);
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
    error_sqr_nn_top(i,1) = sum((table(end-499:end,2)-table(end-499:end,1)).^2)/500; 
    error_sqr_nn_bottom(i,1) = sum((table(1:500,2)-table(1:500,1)).^2)/500;  
    % r_sqr = 1-fraction of variance unexplained (FVU)
    fvu_nn(i,:) = sum((y_test-y_hat_nn).^2)/sum(y_test.^2);
    fvu_nn_top(i,:) = sum((table(end-499:end,2)-table(end-499:end,1)).^2)/sum(y_test.^2);
    fvu_nn_bottom(i,:) = sum((table(1:500,2)-table(1:500,1)).^2)/sum(y_test.^2);  

    

end

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr_nn = (1- sum(fvu_nn)) *100;
r_sqr_nn_top = (1- sum(fvu_nn_top))*100;
r_sqr_nn_bottom = (1- sum(fvu_nn_bottom))*100;

pred_vw_nn = mean(rp_vw_nn,1)'*100;
avg_vw_nn = mean(rp_real_vw_nn,1)'*100;
sd_vw_nn = std(rp_real_vw_nn,1)';
sr_vw_nn = avg_vw_nn./sd_vw_nn/100;

pred_ew_nn = mean(rp_ew_nn,1)'*100;
avg_ew_nn = mean(rp_real_ew_nn,1)'*100;
sd_ew_nn = std(rp_real_ew_nn,1)';
sr_ew_nn = avg_ew_nn./sd_ew_nn/100;

% cumulative returns for value and equal weighted portfolio, L, H , H-L
test_size = size(rp_ew_nn,1);
cumul_ret_nn = zeros(test_size,6);
cumul_ret_nn (1,1:3) = log(1+rp_real_vw_nn(1,[1,end-1,end]));
cumul_ret_nn (1,4:6) = log(1+rp_real_ew_nn(1,[1,end-1,end]));
for t = 1: test_size-1
   cumul_ret_nn (t+1,1:3) = sum(log(1+rp_real_vw_nn(t+1,[1,end-1,end]))) + cumul_ret_nn(t,1:3); 
   cumul_ret_nn (t+1,4:6) = sum(log(1+rp_real_ew_nn(t+1,[1,end-1,end]))) + cumul_ret_nn(t,4:6); 
end

cumul_sorted = sort(cumul_ret_nn);
maxdd_nn = cumul_sorted(end,:) - cumul_sorted(1,:);
