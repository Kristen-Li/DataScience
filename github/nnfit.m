function linet = nnfit(train_set,train_propotion)
    %% split train and test
    x_train = train_set(:,1:end-2);
    y_train = train_set(:,end-1);

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
    [trainInd,valInd,testInd] = divideblock(size(train_set,2),train_propotion,1-train_propotion,0);
    % the first hidden layer use poslin (nelu)
    net.layers{1}.transferFcn = 'poslin';
    net.layers{2}.transferFcn = 'poslin';
    net.layers{3}.transferFcn = 'poslin';
    net.layers{4}.transferFcn = 'poslin';

    % Train the Network
    [net,tr] = train(net,x_train,y_train);%,'useParallel','yes');
    linet=net;

%% results


