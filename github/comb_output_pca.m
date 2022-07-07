i_list = [1:12:61];
T_q1 = [];
T_q10 = [];
nfactors = [];
df = [];

% combine all data
for i = i_list        
    FolderDestination='stocks_output\';
    filename=strcat(FolderDestination,'model_pca_',num2str(i),'.mat');
    file = matfile(filename);
    df = [df;file.result];
    nfactors=[nfactors;file.nfactors];       
end

port_rank = [];
% form portfolios every month, add column port decile
for t = min(df.time_index):max(df.time_index)
    
    
    y_test = df{(df.time_index == t),"eret_test"};
    y_pred = df{(df.time_index == t),"eret_pred"};
    leny = size(y_test,1);
    value = df{(df.time_index == t),"rank_mvel1"}; % save mve before the transformation
    value_next = df{(df.time_index == t+1),"rank_mvel1"};
    equal = ones(size(y_test,1),1);    
    i = t-240;
    [rp_vw_pca(i,:),rp_real_vw_pca(i,:),decile]=r_portfolio_weighted(y_test,y_pred,value);
    [rp_ew_pca(i,:),rp_real_ew_pca(i,:),~]=r_portfolio_weighted(y_test,y_pred,equal);
    port_rank=[port_rank;decile];    

%     turnover_vw(i,:) = sum((1+rp_vw).*(value_next.*q)/sum(value_next.*q)- (1+y_test).*(value.*q)/sum(value.*q));
%     turnover_ew(i,:) = sum((1+rp_eq).*(equal.*q)/sum(equal.*q)- (1+y_test).*(equal.*q)/sum(equal.*q));

end
% calculate stats 
df.decile = port_rank;

%out-of-sample stock-level prediction performance (percentage R_sqared)
r_sqr = 1-sum(df.eret_pred-df.eret_test).^2/sum(df.eret_test).^2;
% r_sqr_pca_top = 1-sum(error_sqr_pca_top)/sum(y_test_pca_top);
% r_sqr_pca_bottom = 1-sum(error_sqr_pca_bottom)/sum(y_test_pca_bottom);

pred_vw_pca = mean(rp_vw_pca,1)'*100;
avg_vw_pca = mean(rp_real_vw_pca,1)'*100;
std_vw_pca = std(rp_real_vw_pca,1)';
sr_vw_pca = avg_vw_pca./100./std_vw_pca*sqrt(12); %annualized sr


pred_ew_pca = mean(rp_ew_pca,1)'*100;
avg_ew_pca = mean(rp_real_ew_pca,1)'*100;
std_ew_pca = std(rp_real_ew_pca,1)';
sr_ew_pca = avg_ew_pca./100./std_ew_pca*sqrt(12); %annualized sr

% cumulative returns for value and equal weighted portfolio, L, H , H-L
test_size = size(rp_real_ew_pca,1);
cumul_ret_pca = zeros(test_size,6);
cumul_ret_pca (1,1:3) = (1+rp_real_vw_pca(1,[1,end-1,end]));
cumul_ret_pca (1,4:6) = (1+rp_real_ew_pca(1,[1,end-1,end]));

start = -inf(1,6);
maxdd = -inf(1,6);
for t = 1: test_size-1
   cumul_ret_pca (t+1,1:3) = ((1+rp_real_vw_pca(t+1,[1,end-1,end]))).*cumul_ret_pca(t,1:3); 
   cumul_ret_pca (t+1,4:6) = ((1+rp_real_ew_pca(t+1,[1,end-1,end]))).* cumul_ret_pca(t,4:6); 
   for j = 1:6
       if cumul_ret_pca(t+1,j) > cumul_ret_pca(t,j)
           start(1,j) = max(start(1,j),cumul_ret_pca(t+1,j));
       else
           % maximum percentage change in cumulative returns
           maxdd(1,j)= max(maxdd(1,j), (start(1,j)-cumul_ret_pca(t+1,j))/(start(1,j)));
       end
   end
  
end

max_loss = min(rp_real_vw_pca);
cumul_sorted = sort(cumul_ret_pca);
maxdd = 100*maxdd;

output = table(pred_ew_pca,avg_ew_pca,std_ew_pca,sr_ew_pca);

clearvars -except output maxdd r_sqr;
FolderDestination='final_results';
filename=strcat('pca_results','.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'-v7.3');