function [rp,rp_real]=r_portfolio_weighted(r_real,r_hat,weight)
% input: the return row at time T*N. could be predicted returns and realized

% r_hat = table(:,2);
% r_real = table(:,1);

q = quantile(r_hat,[0.25, 0.5,0.75]);

v1 = r_hat<=q(1);%return the rows for the first quantile;
v2 = r_hat>q(1) & r_hat<=q(2); 
v3 = r_hat>q(2) & r_hat<=q(3);
v4 = r_hat>q(3);

vw1 = sum(r_hat(v1).*weight(v1)); % forecasted return of the stocks in first quantile 
vw2 = sum(r_hat(v2).*weight(v2));
vw3 = sum(r_hat(v3).*weight(v3));
vw4 = sum(r_hat(v4).*weight(v4)); 
vw5 = vw4-vw1;
rp = [vw1;vw2;vw3;vw4;vw5];


vr1 = r_real<=q(1);%realized return the rows for the first quantile;
vr2 = r_real>q(1) & r_real<=q(2); 
vr3 = r_real>q(2) & r_real<=q(3);
vr4 = r_real>q(3);

vwr1 = sum(r_real(vr1).*weight(vr1)); % the forecasted return of the stocks in first quantile 
vwr2 = sum(r_real(vr2).*weight(vr2));
vwr3 = sum(r_real(vr3).*weight(vr3));
vwr4 = sum(r_real(vr4).*weight(vr4)); 
vwr5 = vwr4-vwr1;
rp_real = [vwr1;vwr2;vwr3;vwr4;vwr5];


