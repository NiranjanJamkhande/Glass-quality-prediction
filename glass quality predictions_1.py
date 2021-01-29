# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:25:01 2020

@author: Admin
"""

#################### Analysis of Glass Quality Prediction ########################

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

train = pd.read_csv(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\Train.csv')
test =  pd.read_csv(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\Test.csv')
sub =   pd.read_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\Sample_Submission.xlsx') 

####
train.corr()

X_train = train.drop(columns = ['class'])
y_train = train['class']
X_test = test

y_train.value_counts()
### checking missing values 

null_col = train.columns[train.isnull().any()]
null_col
train[null_col].isnull().sum()

null_col = test.columns[test.isnull().any()]
null_col
test[null_col].isnull().sum()
## no missing values.

###################################################################################################3
## 1) decision tree
from sklearn.tree import DecisionTreeClassifier


model1 = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=5, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=2020, splitter='best')

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\dtcv.xlsx', index = False)

####
from sklearn.metrics import log_loss
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2020)
cv = GridSearchCV(clf, param_grid=parameters,
                  scoring='neg_log_loss', verbose = 2)

cv.fit(X_train,y_train)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_

################################################
###   2) Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

model1 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features=11,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=2,
                       warm_start=False)


model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\rf_cv.xlsx', index = False)
###

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss

parameters = {'max_features': np.arange(1,15)}


model_rf = RandomForestClassifier(random_state=0)
cv = GridSearchCV(model_rf, param_grid=parameters,
                  scoring='neg_log_loss' , verbose = 2)

cv.fit( X_train , y_train )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)

#################################################################################
####  3)  SGD Classifier ###
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

model1 = SGDClassifier(loss='log',random_state=0, verbose=3)

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\sgd.xlsx', index = False)


########## 4) svc
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

model1 = SVC(probability = True,kernel='rbf',verbose=3)

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\svc.xlsx', index = False)

###
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss


C_range = np.linspace(0.01,4)
#C_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])
gamma_range = np.logspace(-4, 3)
#gamma_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])

parameters = dict(gamma=gamma_range, C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, train_size=2, test_size=None, random_state=42)
svc = SVC(probability=True,verbose=3)


svmGrid = GridSearchCV(svc, param_grid=parameters,
                       scoring='neg_log_loss')
svmGrid.fit(X_train, y_train)

# Best Parameters
print(svmGrid.best_params_)

print(svmGrid.best_score_)


##################################################################################################
### K NN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'n_neighbors': np.array([1,3,5,7,9,11,12,15,17,19,21,23,25])}
print(parameters)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=00)

knn = KNeighborsClassifier()

cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='neg_log_loss')

cv.fit( X_train , y_train )

print(cv.cv_results_  )

# Table of Grid Search CV Results
df_cv = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)
print(cv.best_estimator_)


#####

model1 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=23, p=2,
                     weights='uniform')

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\knn_23.xlsx', index = False)
#################################################################

####  logistic regg.
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\log reg.xlsx', index = False)
####
