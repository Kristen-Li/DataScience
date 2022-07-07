 function opt_k = pcak(k_list,train_sample, train_propotion)
% %load sample.mat;
% train_sample = Z(1:5555,:);
% %train_sample = Z;
% train_propotion = 0.6;
% mark the time line from 1 
train_sample(:,end) = train_sample(:,end) - train_sample(1,end) + 1;
train_window_size = floor(train_sample(end,end)*train_propotion);
test_size = train_sample(end,end)-train_window_size;
msfes = zeros (1,size(k_list,2));

for j = 1:size(k_list,2)
    k = k_list(j);
    train_set = train_sample(train_sample(:,end) >=1 & train_sample(:,end)<= train_window_size -1 + 1,:);
    test_set = train_sample(train_sample(:,end)>= train_window_size+1,:); %maybe include 12 forecasts
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    x_test = test_set(:,1:end-2);
    y_test = test_set(:,end-1);
    %% pca
    [~,Fhat_pca_all,~]=pca([x_train;x_test],'Centered',false);         
    Fhat_pca=Fhat_pca_all(1:end-size(x_test,1),1:k);
    F_pca_new=Fhat_pca_all(end-size(x_test,1)+1:end,1:k);
    %Fit F and Y to get beta
    predictor_pca=[ones(rows(Fhat_pca),1),Fhat_pca];
    betahat_pca=(predictor_pca'*predictor_pca)\predictor_pca'* y_train;
    %Use Fnew and betahat to get Yhat
    y_hat_pca = [ones(size(F_pca_new,1),1),F_pca_new]*betahat_pca;
    % how to save and index the errors? no need 
    % Just check dibold mariano test
    msfe_pca = sum((y_test-y_hat_pca).^2)/size(y_hat_pca,1);        
    msfes(j)=msfe_pca;
end
% find the minimal msfe0
[~,min_index]=min(msfes,[],2);
opt_k = k_list(min_index);