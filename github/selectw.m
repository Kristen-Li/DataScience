function [opt_w,min_MSFE_fixed_k] = selectw(train_sample, train_propotion,optk,w_list)
%load sample.mat;

%train_sample = Z(1:15555,[1:40,end-2:end]);
%optk = 0;
%train_propotion = 0.6;
% mark the time line from 1 

train_sample(:,end) = train_sample(:,end) - train_sample(1,end) + 1;
train_window_size = floor(train_sample(end,end)*train_propotion);
test_size = train_sample(end,end)-train_window_size;
msfes = zeros (1,size(w_list,2));
%% initialize 

%%
for j = 1:size(w_list,2)
    optw = w_list(j);
    train_set = train_sample(train_sample(:,end) >=1 & train_sample(:,end)<= train_window_size -1 + 1,:);
    test_set = train_sample(train_sample(:,end) >= train_window_size+1,:);  %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);

    %% spca       
    [Fhat_spca,A,B]=factorfit(x_train,y_train,optk,optw); %(run x(2:301-5) and y (2+5:301) to get F and beta
    % [~,sdX,meanX]=standard(x_train);
    % Fnew=(x_test-meanX(1,:))./sdX(1,:)*A;     
    F_spca_new = x_test*A;
    predictor_spca=[ones(rows(Fhat_spca),1),Fhat_spca];
    betahat_spca=(predictor_spca'*predictor_spca)\predictor_spca'*y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_spca = [ones(size(F_spca_new,1),1),F_spca_new]*betahat_spca;
    error_sqr_spca = sum((y_test-y_hat_spca).^2)/size(y_hat_spca,1);  
    msfe_spca=sum(error_sqr_spca)/test_size;
    msfes(j)=msfe_spca;
end


[min_MSFE_fixed_k,min_index]=min(msfes,[],2);
opt_w = w_list(min_index);

