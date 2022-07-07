%tunepcak
function runmepcak(jobid)
if jobid == 1
    tune_pca([1 13 25],jobid);
elseif jobid ==2
    tune_pca([37 49 61],jobid);
elseif jobid ==3
    tune_pca([73 85 97],jobid);
elseif jobid ==4
    tune_pca([109,121,133],jobid);
elseif jobid ==5
    tune_pca([145 157 167],jobid);
elseif jobid ==6
    tune_pca([181 193 205],jobid);
elseif jobid ==7
    tune_pca([217 229],jobid);
end
  


