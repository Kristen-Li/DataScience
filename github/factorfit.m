function [Fhat,A,B]=factorfit(X,Y,k,w)%same lengh of x,y. use lambda to forcast fachat, use fachat to forecast yhat
T=size(X,1);
N=size(X,2);
%[XS,~,~]=standard(X);
% [YS,~,~]=standard(Y);
XS=X;
YS=Y;
%P=X*pinv(X'*X)*X';
P=sqrt(N);
% if T<N
%     D=w*((X*X'))+(1-w)*((Z*Z'));
%     [Fhat0,~,~]=svd(D);
%     Fhat=Fhat0(:,1:k);
%     lambda=((Fhat'*Fhat)\Fhat'*X)';
% else
%     D=w*(X'*X)+(1-w)*(Z'*Z);
%         
%     [lambda0,~,~]=svd(D'*D);
%     lambda=lambda0(:,1:k);
%     Fhat=X*lambda/(lambda'*lambda);
%     lambda=((Fhat'*Fhat)\Fhat'*X)';
% end
C=sqrt(w)*XS;
D=sqrt(1-w)*YS;

if w==1 
    D=[];
end
 [~,Fhatall]=pca([C D]);  %not centered or standardized

 Fhat=Fhatall(:,1:k);
% predictor=[ones(rows(Fhat),1),Fhat];
   lambda=((Fhat'*Fhat)\Fhat'*[XS YS])'; %you missed a \sqrt w
%lambda= ((predictor'*predictor)\predictor'*[X Y])';
lambda=lambda(1:end-1,:);
 %X=Fhat*lambda'+e; F is T*k, lambda is N*k
 
%  given new X, what't the new F, F=X*A ?;A is N*k
 A=pinv(XS)*Fhat; %eq1 when T<N eq2,3 are better, when T>N pinv is better. 
%A=pinv(X'*X)*X'*predictor;

 %A=lambda/(lambda'*lambda);  % eq2 N*k*k*k     (lambda*lambda')\lambda N*N*N*k
%A=pinv(lambda*lambda')*lambda; %eq3, 2 and 3 are the same. 

%[~,F0]=pca(X);
%A=pinv(F0'*F0)*F0'*Fhat;
 B=lambda(end,:);












%chatx=Fhat*lambda'; chatz=Fhat*betahat;