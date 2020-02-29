## Mistakes
+ After hyparam optimization, doesn't use validation set for training
+ `random_state` among hyparams in SVM. Values for `C` also look suspicious.
+ `optimalK`: Gap seems to increase with higher no of clusters. Also, if best value is `k`, its returning `k-1`, that's why I believe best no of clusters in paper is 13.
