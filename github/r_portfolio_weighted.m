function [rp,rp_real,decile]=r_portfolio_weighted(r_real,r_hat,value)
% input: the return row at time T*N. could be predicted returns and realized
% 
%  r_hat = y_hat_spca;
%  r_real = y_test;
%  value = equal;

q = quantile(r_hat,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]);

v1 = r_hat<=q(1);%return the rows for the first quantile;
v2 = r_hat>q(1) & r_hat<=q(2); 
v3 = r_hat>q(2) & r_hat<=q(3);
v4 = r_hat>q(3) & r_hat<=q(4);
v5 = r_hat>q(4) & r_hat<=q(5);
v6 = r_hat>q(5) & r_hat<=q(6);
v7 = r_hat>q(6) & r_hat<=q(7);
v8 = r_hat>q(7) & r_hat<=q(8);
v9 = r_hat>q(8) & r_hat<=q(9);
v10 = r_hat>q(9);


weight1=value(v1)./sum(value(v1));
weight2 =value(v2)./sum(value(v2));
weight3=value(v3)./sum(value(v3));
weight4=value(v4)./sum(value(v4));
weight5=value(v5)./sum(value(v5));
weight6=value(v6)./sum(value(v6));
weight7=value(v7)./sum(value(v7));
weight8=value(v8)./sum(value(v8));
weight9=value(v9)./sum(value(v9));
weight10=value(v10)./sum(value(v10));

vw1 = sum(r_hat(v1).* weight1); % forecasted return of the stocks in first quantile 
vw2 = sum(r_hat(v2).* weight2);
vw3 = sum(r_hat(v3).* weight3);
vw4 = sum(r_hat(v4).* weight4); 
vw5 = sum(r_hat(v5).* weight5); 
vw6 = sum(r_hat(v6).*weight6); 
vw7 = sum(r_hat(v7).*weight7); 
vw8 = sum(r_hat(v8).*weight8); 
vw9 = sum(r_hat(v9).*weight9); 
vw10 = sum(r_hat(v10).*weight10); 
vw11 = vw10-vw1;
rp = [vw1;vw2;vw3;vw4;vw5;vw6;vw7;vw8;vw9;vw10;vw11];


r_hat_11 = [-r_hat(v1);r_hat(v10)];
weight11 = [0.5*weight1;0.5*weight10];

vwr1 = sum(r_real(v1).*weight1); % realized return of the stocks in first quantile 
vwr2 = sum(r_real(v2).*weight2);
vwr3 = sum(r_real(v3).*weight3);
vwr4 = sum(r_real(v4).*weight4); 
vwr5 = sum(r_real(v5).*weight5); 
vwr6 = sum(r_real(v6).*weight6); 
vwr7 = sum(r_real(v7).*weight7); 
vwr8 = sum(r_real(v8).*weight8); 
vwr9 = sum(r_real(v9).*weight9); 
vwr10 = sum(r_real(v10).*weight10); 
vwr11 = vwr10-vwr1;
rp_real = [vwr1;vwr2;vwr3;vwr4;vwr5;vwr6;vwr7;vwr8;vwr9;vwr10;vwr11];

decile = ones(size(r_real,1),1);
decile(v2)=2;
decile(v3)=3;
decile(v4)=4;
decile(v5)=5;
decile(v6)=6;
decile(v7)=7;
decile(v8) = 8;
decile(v9)=9;
decile(v10)=10;

r_real_11 = [-r_real(v1);r_real(v10)];
weight11 = [0.5*weight1;0.5*weight10];



