%%%% read data %%%%
dstemp = datastore('datashare.csv','TreatAsMissing','NA');
dstemp.VariableNames';
colnames =  dstemp.VariableNames';
% the stocks have no missing features. 
% dstemp.SelectedVariableNames={'permno','DATE','RET','prc','target','dp','ep_mac',	'bm_mac',	'ntis',	'tbl',	'tms','dfy','svar',	'time_index'};
dstemp.SelectedVariableNames={'permno','DATE','cfp'};
Ttemp = tall(dstemp);
Dtemp=gather(Ttemp);
Dtemp(Dtemp.permno == 37751,:);


ds = datastore('datashare.csv','TreatAsMissing','NA');
T=tall(ds);
D=gather(T);

Dtest = outerjoin (D,Dtemp,'Type','left');
idx= any(isnan(Dtest.permno_Dtemp),2);
X = Dtest(~idx,:);

D=sortrows(D,{'permno','DATE'},{'ascend','ascend'}); 
ds.SelectedVariableNames={'absacc','acc','aeavol','age','agr','baspread','beta','betasq','bm','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia','chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','ep','gma','grcapx','grltnoa','herf','hire','idiovol','ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m','mom6m','ms','mve_m'};
T1 = tall(ds);
ds.SelectedVariableNames = {'mve_ia','nincr','operprof','orgcap','pchcapx','pchcurrat','pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale','realestate','retvol','roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','std_dolvol','std_turn','stdacc','stdcf','tang','tb','turn','zerotrade'};
T2 = tall(ds);
ds.SelectedVariableNames = {'DATE','permno','sic2','RET','prc','target','dp','ep_mac',	'bm_mac',	'ntis',	'tbl',	'tms','dfy','svar',	'time_index'};
T3 = tall(ds);
D1 = gather(T1);
D2 = gather(T2);
D3 = gather(T3);
dummy = D3.sic2;
sic = dummyvar(dummy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add sic before deleting any rows
D = [D1 D2 D3(:,[1:3,5:6])];
% check the columns with more than 15% missing values
D=D(:,(sum(ismissing(D),1))<size(D,1)*0.15);
sic = sic(~any(ismissing(D),2),:);
D=D(~any(ismissing(D),2),:);
%%%%%creat date variable and create target values %%%%%
D.DATE=num2str(D.DATE);
DATE_YM = str2double(string(D.DATE(:,1:6)));
DATE_YEAR =str2double(string(D.DATE(:,1:4)));
DATE_MONTH = str2double(string(D.DATE(:,5:6)));
DATE_DAY = str2double(string(D.DATE(:,7:8)));
D.date_time = datetime(DATE_YEAR,DATE_MONTH,DATE_DAY);
D.ym = DATE_YM;

%% read risk free return
[~, ~, raw] = xlsread('C:\Users\mjjan\Box\Stocks\data\rf.xlsx','Sheet1','A2:B1129');
data = reshape([raw{:}],size(raw));
rf = table;
rf.RF = data(:,1);
rf.ym = data(:,2);
clearvars data raw;
%%%%%%%%%%%calculate risk premium%%%%%%%%%%%%%%%%
D = join(D,rf);
D.RET = D.RET -D.RF;
%%%%%%%%%%%%%%%%% add one month ahead target risk premium
% sort D1 by equity then date. 
D=X;
 D=sortrows(D,{'permno_D','DATE_D'},{'ascend','ascend'}); 
 %shift target by one month % change!
 D.target = [D.RET(2:end,:);NaN(1,1)];
 for i = 1: size(D,1)-1
    if D.permno_D(i,:) ~= D.permno_D(i+1,:)      
        D.target(i)= NaN(1,1);
    end
 end
 idx= any(isnan(D.target),2);
 sum(idx)
 sic = sic(~any(ismissing(D),2),:);
 D=D(~any(ismissing(D),2),:);
% resort D1 by date then equity. 
 D=sortrows(D,{'date_time','permno'},{'ascend','ascend'});
% get year, month and day info
[D.y,D.m,D.d] = ymd(D.date_time);
key = D.permno;
save('key',D.permno);


%% generate index for different months, all together 480months %%%%
time_index = ones(size(D,1),1);
for i = 2 : size(D,1)
    if D.DATE(i,:) == D.DATE(i-1,:)
        time_index(i) = time_index(i-1);
    else 
        time_index(i) = time_index(i-1) + 1;
    end
end
D.time_index = time_index;

%% combine stock-level and macro index plus a constance, normalized data%%%%
%Z = zeros(1,(size(D1,2)-1)*size(macro,2));
%import macro variables
%Import the data
[~, ~, raw] = xlsread('PredictorData2019.xlsx','8 macro','A2:I505');
% Create output variable
macro = reshape([raw{:}],size(raw));
% delete na rows DONE
% add macros
% add time, gvs, dummies
Z2 = [];
for t = 1:480
   stock_level = D(D.time_index == t, 1:79); % first 79 cols are stock charecteristics
   macro_level = macro(t,2:end);
   % for each time, for each charecteristic, rank the stock to [-0.5,0.5];
   stock_ranked = zeros(size(stock_level,1),size(stock_level,2));
   for char = 1:size(stock_level,2)
       [~,ranked]  = ismember(stock_level(:,char),unique(stock_level(:,char)));
       stock_ranked(:,char) = ranked / size(ranked,1) - 0.5;
   end
   temp_ranked = kron(stock_ranked, macro_level); 

   Z2 = [Z2;temp_ranked];
end
%% add sector dummies. Also add target, and time index back

%D.date_time=[];
%variable_names = D.Properties.VariableNames;
%variable_names = variable_names';
sic = sic(:,mean(sic)~=0);
Z = [Z2,sic,D.target, D.time_index];
save('sample.mat', 'Z');
% % 
% 
% 
% % look through all the rolling windows
% cleaned_df = D1(:,[3,4,6,7]); 
% tstPerform = zeros(1,240);
% parfor i = 1
%     i
%     train_set = cleaned_df (cleaned_df.time_index >=i & cleaned_df.time_index<= 239 + i,:);
%     test_set = cleaned_df(cleaned_df.time_index == 240+i,:);
%     
%     % neural network test
%     net.trainParam.showWindow = false;
%     net.performFcn = 'mse';  % Mean Squared Error
% 
%     train_set= train_set{:,:}';
%     test_set = test_set{:,:}';
%     x = train_set(2:3,:);
%     t = train_set(1,:);
%     x_test = test_set(2:3,:);
%     t_test = test_set(1,:);
%     % Choose a Training Function
%     % For a list of all training functions type: help nntrain
%     % 'trainlm' is usually fastest.
%     % 'trainbr' takes longer but may be better for challenging problems.
%     % 'trainscg' uses less memory. Suitable in low memory situations.
%     trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% 
%     % Create a Fitting Network
%     hiddenLayerSize = 10;
%     net = fitnet(hiddenLayerSize,trainFcn);
% 
%     % Setup Division of Data for Training, Validation, Testing 
%     [trainInd,valInd,testInd] = divideblock(size(train_set,2),0.6,0.4,0);
% 
%     % Train the Network
%     [net,tr] = train(net,x,t);
% 
%     % the overall performance of the trained Network
%     y = net(x);
%     e = gsubtract(t,y);
%     performance = perform(net,t,y);
% 
%     % performance only on the test set.
%     tstOutputs = net(x_test);
%     tstPerform(i) = perform(net, t_test, tstOutputs);
% end;
% 
% 
% 
% 
