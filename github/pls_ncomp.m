function  opt_ncomp = pls_ncomp(n_list,train_sample,train_propotion)
% %load sample.mat;
% train_sample = Z(1:5555,:);
% %train_sample = Z;
% train_propotion = 0.6;
% mark the time line from 1 
train_sample(:,end) = train_sample(:,end) - train_sample(1,end) + 1;
train_window_size = floor(train_sample(end,end)*train_propotion);
test_size = train_sample(end,end)-train_window_size;
error_sqr_pls = zeros(test_size,1);
msfes = zeros (1,size(n_list,2));

for j = 1:size(n_list,2)
    ncomp = n_list(j);
    for i = 1:test_size
        train_set = train_sample(train_sample(:,end) >=i & train_sample(:,end)<= train_window_size -1 + i,:);
        test_set = train_sample(train_sample(:,end)== train_window_size+i,:); %maybe include 12 forecasts
        %% split train and test
        x_train = train_set(:,1:end-2);
        y_train = train_set(:,end-1);
        x_test = test_set(:,1:end-2);
        y_test = test_set(:,end-1);
        %% pls
        [XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(x_train,y_train,ncomp);
        [~,npls]=min(MSE(2,:));
        [XL,YL,XS,YS,betapls,PCTVAR,MSE] = plsregress(x_train,y_train,max(3,npls-1));
        y_hat_pls = [ones(size(x_test,1),1),x_test]*betapls;
        error_sqr_pls(i,1) = sum((y_test-y_hat_pls).^2)/size(y_hat_pls,1);        
    end;
    msfe_pls=sum(error_sqr_pls)/test_size;
    msfes(j)=msfe_pls;
end
% find the minimal msfe0
[~,min_index]=min(msfes,[],2);
opt_ncomp = n_list(min_index); 
  