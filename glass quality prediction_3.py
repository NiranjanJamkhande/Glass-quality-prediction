# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:00:09 2020

@author: Admin
"""

### Voting


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



dtc = DecisionTreeClassifier(random_state=0)
logreg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)


from sklearn.ensemble import VotingClassifier
model1 = VotingClassifier(estimators=[('DT',dtc),
                                      ('LR',logreg),
                                      ('KNN',knn)],
                                             voting='soft')


model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\voting1.xlsx', index = False)

#######################################################

#### stacking


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict_proba(X_test)









from sklearn.svm import SVC
svc = SVC(probability = True,kernel='linear')

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42,max_depth = 5)


models_considered = [('Logistic Regression', logreg),
                     ('SVC', svc),
                     ('Decision Tree',dtc)]

#from xgboost import XGBClassifier
#clf = XGBClassifier(random_state=42)


from sklearn.tree import DecisionTreeClassifier
xyz = DecisionTreeClassifier(random_state=42)

from sklearn.ensemble import StackingClassifier
model1 = StackingClassifier(estimators = models_considered,
                           final_estimator=xyz,stack_method="predict_proba",
                           passthrough=True,verbose=5)

#model = CalibratedClassifierCV(model1)
model1.fit(X_train, y_train)

submission = model1.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\voting1.xlsx', index = False)

### Ada boost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model1 = AdaBoostClassifier(
                                 DecisionTreeClassifier(max_depth=5),
                                  n_estimators=200
                                )

model = CalibratedClassifierCV(model1)
model.fit(X_train, y_train)

submission = model.predict_proba(X_test)
submission = pd.DataFrame(submission)
submission = submission.rename(columns = {0:1 , 1:2 })


submission.to_excel(r'C:\Users\Admin\Desktop\Glass_Quality_Predictions_Data\ada boost.xlsx', index = False)

####