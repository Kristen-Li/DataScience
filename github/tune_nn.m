function tune_nn(i_list,jobid)
%load samplez.mat;
%load part of the full file
eO = load_mat('samplez.mat');
train_window_size = 240;
z = eO.Z(:,end);   
%%
for i = i_list
    
   % train_set = Z(Z(:,end) >=i & Z(:,end)<= train_window_size -1 + i,:); % subsample 100 features 
    a = (z>=i);
    b = (z <= train_window_size -1+i);
    train_ind = find(z(a&b));
    train_set = eO.Z(min(train_ind):max(train_ind),:);   
    
     %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);
    
   %% neural network test
    
    net.performFcn = 'mse';  % Mean Squared Error

    x_train = x_train';
    y_train = y_train';
  
    % Create a Fitting Network
     hiddenLayer1Size = 32;
     hiddenLayer2Size = 16;
     hiddenLayer3Size = 8;
     hiddenLayer4Size = 4;
    
     net = fitnet([hiddenLayer1Size hiddenLayer2Size hiddenLayer3Size hiddenLayer4Size],  trainFcn);
     net.trainFcn = 'traingdx';
     net.trainParam.epochs = 100;
     net.trainParam.lr = 0.01;	


    % Setup Division of Data for Training, Validation, Testing 
    [trainInd,valInd,testInd] = divideblock(size(train_set,2),0.7,0.3,0);
    % the first hidden layer use poslin (nelu)
     net.layers{1}.transferFcn = 'poslin';
     net.layers{2}.transferFcn = 'poslin';
     net.layers{3}.transferFcn = 'poslin';
     net.layers{4}.transferFcn = 'poslin';

    % Train the Network
    [net,tr] = train(net,x_train,y_train);%,'useParallel','yes');
    linet=net;
end
%% results

mkdir('st_output')
FolderDestination='stocks_output';
filename = strcat('net_',int2str(jobid),'.mat');
matfile = fullfile(FolderDestination, filename);
save(matfile,'linet');

