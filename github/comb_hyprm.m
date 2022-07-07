ncomp_spca = [];
for i = [1 2 3 4 5 6 7 8 9 10]
    i
    filename = strcat('kw_spca_',num2str(i),'.mat');
    load(filename)
    ncomp_spca = [ncomp_spca;k_tune];
end
save('ncomp_spca','ncomp_spca')
