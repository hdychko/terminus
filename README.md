# Terminus

1. Selected features ([Experian - Data Analysis](https://apsgrp-my.sharepoint.com/:x:/g/personal/halyna_dychko_cashplus_com/EaLINnf4Kx9DuKQWYAAI8G4B_hnVSlTNUWlmq24YbL3SjQ?e=hwlxyk)) based on the [AIQ2 Block - Cashplus.csv](https://apsgrp-my.sharepoint.com/:x:/g/personal/james_coveney_cashplus_com/Ecug3rMiXjJApvfxQmHeh4sBkFJwgq5hGboj2FB80kB2jA?e=2K6Skl) and [Final_Dataset.csv](https://apsgrp-my.sharepoint.com/:x:/g/personal/james_coveney_cashplus_com/EeHpbiMrUzZGnxQGimFWQysBG8tfUYfW5nX0NI0Wx5oAHg?e=LcWJxg):     
    
    A. [[EDA][Feature Selection] Final_AIQ2.ipynb](https://github.com/hdychko/terminus/blob/master/notebooks/%5BEDA%5D%5BFeature%20Selection%5D%20Final_AIQ2.ipynb)    
    
    B. [[EDA][Feature Binning] Final_AIQ2.ipynb](https://github.com/hdychko/terminus/blob/master/notebooks/%5BEDA%5D%5BFeature%20Binning%5D%20Final_AIQ2.ipynb) - final set of 50 features    

2. Modelling

    A. [Experian][Initial]
   
    Logistic Regression (LR) is trained on 50 features preselected with the procedure involving correlation, optimal binning (the output of 1.B). It's retrained iteratively until all coefficients are greater than 0 and their p-value > 5% confidence level.
    
    Random Search (RS) technique is used to search for the optimal hyperparameters values. The candidates are randomly generated from the distributions/choices below: 
    
    ```
        penalty=['l2', 'l1', 'elasticnet'], 
        C=uniform(loc=0, scale=20), 
        fit_intercept=[True], 
        class_weight=['balanced'], 
        random_state=[42], 
        n_jobs=[-1]
    ```
    
    TRAIN: [`2021-12-01` , `2022-01-01`)
    VALIDATION: [`2022-01-01`, `2022-05-01`)
    TEST: [`2022-05-01` , `2022-08-01`)
    
    *The last date is not included in each interval provided above.
    
    **Experiments**
   
    **01-LogRegression-hyperparams_tunning** - RS for LR with Target: 
    1 - where `GB6_Flag_2Limit` = 'B'
    0 - where (`GB6_Flag_2Limit` = 'G') or (`GB6_Flag_2Limit` = 'I')
    
    **02-LogRegression-hyperparams_tunning** - continuation of `01-LogRegression-hyperparams_tunning`, excluding features with ecoefficiencies of LR equal 0.
    
    **03-LogRegression-hyperparams_tunning** - RS for LR with the train and validation datasets included only observations with (GB6_Flag_2Limit` = 'B') or (`GB6_Flag_2Limit` = 'G'). 
    
    **04-LogRegression-hyperparams_tunning** - continuation of `03-LogRegression-hyperparams_tunning` excluding features with coefficients = 0 and/or p-value > 5%.

    B. [Experian][RF-based features]

