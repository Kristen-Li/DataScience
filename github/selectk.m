function [optweight,optk]=selectk(k_list,w_list,train_sample,train_propotion)
kmax = size(k_list,2);
optweightk=zeros(1,kmax);
minMSFEk=zeros(1,kmax);
parfor i=1:kmax
   [ optweightk(1,i),minMSFEk(1,i)]=selectw(train_sample, train_propotion,k_list(1,i),w_list); %optimal weight and ratio as a function of k
end
[minMSFE,optk]=min(minMSFEk,[],2);
optweight=optweightk(:,optk);