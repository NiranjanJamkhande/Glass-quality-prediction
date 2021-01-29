# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:17:48 2020

@author: Admin
"""
### Bagging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

model1 = BaggingClassifier(  base_estimator = RandomForestClassifier(random_state=1211,
                                  n_estimators=50,oob_score=True),
                            n_estimators=50,
                             random_state=1211,oob_score=True,
                             max_features=X_train.shape[1],
                             max_samples=500,verbose=2)
                             

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\bagg.xlsx', index = False)

###################################################################
###   Xgboost
from xgboost import XGBClassifier


model1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.2, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=2020,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=3)
                             

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\xgb_cv.xlsx', index = False)
####Tunning using Randomized Search CV ############
lr_range = [0.001,0.01,0.2,0.5,0.6,1]
depth_range = [3,4,5,6,7,8,9]

parameters = dict(learning_rate=lr_range,
                  max_depth=depth_range)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2020)

from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV
clf = XGBClassifier(random_state=2020)
rcv = RandomizedSearchCV(clf, param_distributions=parameters,
                  cv=kfold,scoring='neg_log_loss',n_iter=15,random_state=2020,verbose = 2)

rcv.fit(X_train,y_train)
df_rcv = pd.DataFrame(rcv.cv_results_)
print(rcv.best_params_)

print(rcv.best_score_)
rcv.best_estimator_
##########################
###  Voting

