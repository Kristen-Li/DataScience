clear 
clc

d=5; % 10 %20 %30 %50number of stocks
r=3; % number of common factors
t= 5/252 ;   %window size =dt one week
step=390*12; % 5s, 1minute 390, 390/5
dt=t/step;
kn=20;%0.5*(dt)^(-0.5)*sqrt(log(d));
simulation=100;
%parameters
k=[3;4;5]; theta=[0.05;0.04;0.03]; eta=[0.3;0.4;0.3]; rho=[-0.6;-0.4;-0.25]; mu=[0.05;0.03;0.02];
kt=[1;2;3;];thetat=[unifrnd(0.25,1.75,[d,1]),0.5*randn(d,1),0.5*randn(d,1)]; 
epsilont=[0.5;0.6;0.7];
lambdaF=1/t; muF=4*sqrt(dt);lambdaZ=2/t; muZ=6*sqrt(dt); muS=sqrt(dt);
kappa=4; thetascaler=0.3; etascaler=0.06;
% X=zeros(d,step+1); sigma=sqrt(theta).*ones(r,step+1);
% sigmasq=theta.*ones(r,step+1); beta=zeros(d,r,step+1); F=zeros(r,step+1);
% Z=zeros(d,step+1); gamma=zeros(step+1,1); gammasq=zeros(step+1,1);
Vsimple=zeros(d,r,simulation);    V4=zeros(d,1,simulation); f0simple=zeros(d,r,simulation); f4=zeros(d,1,simulation);
eigvlsimple=zeros(r,simulation);    eigvl4=zeros(1,simulation); eigvls0=zeros(r,simulation); eigvl40=zeros(1,simulation);
covtrue=zeros(d,d);
BPV=zeros(d,1); dxcol=zeros(d,step);
beta(:,:,1)=thetat; 
eigvlcorrected=zeros(d,simulation);
first=zeros(simulation,1); second=zeros(simulation,1); third=zeros(simulation,1); fourth=zeros(simulation,1);
firstb=zeros(simulation,1); secondb=zeros(simulation,1); thirdb=zeros(simulation,1); fourthb=zeros(simulation,1);

for m=1:simulation
    m
% generate stochastic process
    cov=[1,0.05,0.1,-0.6,0,0;
        0.05,1,0.15,0,-0.4,0;
        0.1,0.15,1,0,0,-0.25;
        -0.6,0,0,1,0,0;
        0,-0.4,0,0,1,0;
        0,0,-0.25,0,0,1];
    sixw=sqrt(dt)*mvnrnd(zeros(6,1),cov,step); %w1;w2;w3; wtilde1;wtilde2;wtilde3
    sixw=sixw';
    dW=sixw(1:3,:);
    dWtilde=sixw(4:6,:);
    
    dBhat=sqrt(dt)*randn(step,1);
    dBtilde=sqrt(dt)*randn(d,r,step);
    % jump up 50% time, and down 50%time , jump down higher prob
    signF=unidrnd(2,3,step); % discret uniform [1,2]
    signF(signF==2)=-1;
    signZ=unidrnd(2,3,step);
    signZ(signZ==2)=-1;
    poissonF=poissrnd(lambdaF*dt,3,step);
%     dJF = poissonF.*(random('exp',muF,3,step)).*signF;
%     dJZ= poissrnd(lambdaZ*dt,3,step).*(random('exp',muZ,3,step)).*signZ;
%     dJS=poissonF.*(random('exp',muS,3,step));
    dJF = zeros(3,step);
    dJZ= zeros(d,step);
    dJS=zeros(3,step);

 X=zeros(d,step+1); sigma=sqrt([1;1;1].*theta).*ones(r,step+1); sigmasq=[1;1;1].*theta.*ones(r,step+1); beta=zeros(d,r,step+1); F=zeros(r,step+1); Z=zeros(d,step+1);
%gamma=0.001-0.002*rand(step+1,1); gammasq=0.1*ones(step+1,1);
gamma=zeros(step+1,1); gammasq=zeros(step+1,1);
covtrue=zeros(d,d);
 BPV=zeros(d,1); dxcol=zeros(d,step);
beta(:,:,1)=thetat; 
for s=1:step
 
      sigmasq(:,s+1)=sigmasq(:,s)+k.*(theta-sigmasq(:,s))*dt+eta.*sqrt(sigmasq(:,s)).*dWtilde(:,s)+dJS(:,s);
     sigma(:,s+1)=sqrt(sigmasq(:,s+1));
  
    
   % gammasq(s)=gamma(s).*gamma(s);
  %  gammasq(s+1)=gammasq(s)+kappa*(0.01-gammasq(s))*dt+etascaler*gamma(s)*dBhat(s);
   % %gamma(s+1)=gamma(s)+kappa*(thetascaler-gamma(s))*dt+etascaler*dBhat(s);
    %gamma(s+1)=sqrt((gammasq(s+1)));
    beta(:,1,s+1)=beta(:,1,s) +kt(1)*(thetat(:,1)-beta(:,1,s))*dt+epsilont(1)*sqrt(beta(:,1,s)).*dBtilde(:,1,s);
    beta(:,2:r,s+1)=beta(:,2:r,s) +repmat(kt(2:3)',d,1).*(thetat(:,2:r)-beta(:,2:r,s))*dt+repmat(epsilont(2:r)',d,1).*dBtilde(:,2:r,s);
%      beta(:,2,s+1)=beta(:,2,s) +kt(2)*(thetat(:,2)-beta(:,2,s))*dt+epsilont(2)*sqrt(beta(:,2,s)).*dBtilde(:,2,s);
%     beta(:,[1 3],s+1)=beta(:,[1 3],s) +repmat(kt([1 3])',d,1).*(thetat(:,[1 3])-beta(:,[1 3],s))*dt+repmat([0.5 0.7],d,1).*dBtilde(:,[1 3],s);
%    
    F(:,s+1)=F(:,s)+mu*dt+sigma(s)*sqrt(dt)*randn(r,1)+dJF(s); 
    Z(:,s+1)=Z(:,s)+gamma(s)*sqrt(dt)*randn(d,1)+dJZ(s);
    X(:,s+1)= X(:,s)+beta(:,:,s)*(F(:,s+1)-F(:,s))+(Z(:,s+1)-Z(:,s)); % Beta is dxr, F is rx1
    dxcol(:,s)=X(:,s+1)-X(:,s);
  
 covtrue=beta(:,:,s)*sigma(s)*sigma(s)'*beta(:,:,s)'+gamma(s)*gamma(s);
 %covtrue=dxcol(:,s)*dxcol(:,s)';
  %covtrue=covtrue/dt;
     [f0,eigvltrue] = eig(covtrue);
     eigvltrue=diag(eigvltrue);
    [eigvltrue,INDEX] = sort(eigvltrue,'descend');
     f0 = f0(:, INDEX);
     f0simple=f0simple+f0(:,1:3)*dt;
     f4=f4+mean(f0(:,4:end),2)*dt;
     eigvls0(:,m)=eigvls0(:,m)+eigvltrue(1:3)*dt;
     eigvl40(m)=eigvl40(m)+mean(eigvltrue(4:end))*dt;
  
    
    
    if step>=2
        %BPV(1,1)
    BPV=BPV+(abs(X(:,step)-X(:,step-1)).*abs(X(:,step+1)-X(:,step)))*pi/2;
    end
   
end    
 un=1.7*10^(-4)+3*sqrt(BPV)*dt^0.47;
for i =1: t/(kn*dt) 
     %estimated cov
    
    dX=dxcol(:,i*kn-(kn-1):i*kn);
  %dX(abs(dX)>=un)=0;
   
    covhat=(dX*dX')/(kn*dt);
    [V0,eigvl] = eig(covhat);
    eigvl=diag(eigvl);
   [eigvl,I] = sort(eigvl,'descend');
   V0 = V0(:, I);
    Vsimple(:,:,m)=Vsimple(:,:,m)+V0(:,1:3)*kn*dt;
    V4(:,m)=V4(:,m)+mean(V0(:,4:end),2)*kn*dt;
    eigvlsimple(:,m)=eigvlsimple(:,m)+eigvl(1:3)*kn*dt;
    eigvl4(m)=eigvl4(m)+mean(eigvl(4:end))*kn*dt;
  %bias corrected 
  Vcorrected=zeros(length(eigvl),1);
  for g=1:length(eigvl)
  %Vcorrected(g)=eigvl(g)-1/kn*trace(max((eigvl(g)-covhat),0)*covhat)*eigvl(g);
    Vcorrected(g)=eigvl(g)-1/kn*trace(pinv(eigvl(g)-covhat)*covhat)*eigvl(g);

  end
  eigvlcorrected(:,m)= eigvlcorrected(:,m)+Vcorrected*kn*dt;
 
end
   first(m)=1/sqrt(dt)*(eigvlcorrected(1,m)-eigvls0(1,m)) ;
   second(m)=1/sqrt(dt)*(eigvlcorrected(2,m)-eigvls0(2,m)) ;
   third(m)=1/sqrt(dt)*(eigvlcorrected(3,m)-eigvls0(3,m)) ;
   fourth(m)=1/sqrt(dt)*(eigvlcorrected(4,m)-eigvl40(m)) ;
   firstb(m)=1/sqrt(dt)*(eigvlsimple(1,m)-eigvls0(1,m)) ;
   secondb(m)=1/sqrt(dt)*(eigvlsimple(2,m)-eigvls0(2,m)) ;
   thirdb(m)=1/sqrt(dt)*(eigvlsimple(3,m)-eigvls0(3,m)) ;
   fourthb(m)=1/sqrt(dt)*(eigvl4(m)-eigvl40(m)) ;
end  
 mu = 0; sig = 1;
x = linspace (mu-4*sig, mu+4*sig); 

[f1,x1]=hist(first/std(first),50);
h1=bar(x1,f1/trapz(x1,f1),'hist'); 
h1.FaceColor='Magenta';
set(gca,'XLim',[-4,4]);
set(gca,'YLim',[0,0.5]);
hold on;
[f1b,x1b]=hist(firstb/std(firstb),50);
h2=bar(x1b,f1b/trapz(x1b,f1b));
h2.FaceColor='none';
h2.LineStyle='--';
 plot(x, normpdf (x,mu,sig));
hold off;

savefig('first.fig')

[f1,x1]=hist(second/std(second),50);
h1=bar(x1,f1/trapz(x1,f1),'hist'); 
h1.FaceColor='Magenta';
set(gca,'XLim',[-4,4]);
set(gca,'YLim',[0,0.5]);
hold on;
[f1b,x1b]=hist(secondb/std(secondb),50);
h2=bar(x1b,f1b/trapz(x1b,f1b));
h2.FaceColor='none';
h2.LineStyle='--';
 plot(x, normpdf (x,mu,sig));
hold off;

savefig('second.fig')

[f1,x1]=hist(third/std(third),50);
h1=bar(x1,f1/trapz(x1,f1),'hist'); 
h1.FaceColor='Magenta';
set(gca,'XLim',[-4,4]);
set(gca,'YLim',[0,0.5]);
hold on;
[f1b,x1b]=hist(thirdb/std(thirdb),50);
h2=bar(x1b,f1b/trapz(x1b,f1b));
h2.FaceColor='none';
h2.LineStyle='--';
 plot(x, normpdf (x,mu,sig));
hold off;

savefig('third.fig')
% 
[f1,x1]=hist(fourth/std(fourth),50);
h1=bar(x1,f1/trapz(x1,f1),'hist'); 
h1.FaceColor='Magenta';
set(gca,'XLim',[-4,4]);
set(gca,'YLim',[0,0.5]);
hold on;
[f1b,x1b]=hist(fourthb/std(fourthb),50);
h2=bar(x1b,f1b/trapz(x1b,f1b));
h2.FaceColor='none';
h2.LineStyle='--';
 plot(x, normpdf (x,mu,sig));
hold off;
% 
% savefig('fourth.fig')
format long
eigvalue=1/sqrt(dt)*mean([eigvlsimple;eigvl4],2);
correigvalue=1/sqrt(dt)*mean([eigvlcorrected(1:3,:);mean(eigvlcorrected(4:end,:))],2);
eigvaluetrue=1/sqrt(dt)*mean([eigvls0;eigvl40],2)
bias1=mean(first)*sqrt(dt);
bias2=mean(second)*sqrt(dt);
bias3=mean(third)*sqrt(dt);
bias4=mean(fourth)*sqrt(dt);
stdev1=std(first*sqrt(dt));
stdev2=std(second*sqrt(dt));
stdev3=std((third)*sqrt(dt));
stdev4=std((fourth)*sqrt(dt));
bias=[bias1;bias2;bias3;bias4]
stdev=[stdev1;stdev2;stdev3;stdev4]
a1=eigvaluetrue;
a2=bias;
a3=stdev;

save('tablefsfst.mat','eigvaluetrue','bias','stdev')
%change stock 