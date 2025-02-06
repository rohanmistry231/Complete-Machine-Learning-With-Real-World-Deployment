# Loan Approval Prediction Model

## Importing the libraries and dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('train.csv')

df.head()

df.tail()

df.info()

df.describe()

df.isnull().sum()

"""## Data Visualization"""

plt.figure(figsize=(7,5))
sns.countplot(df["Loan_Status"])
plt.title("Count of Loan Application Outcomes")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Gender")
plt.title("Count of Loan Application Outcomes based on Gender")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Married")
plt.title("Count of Loan Application Outcomes based on Marital Status")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Dependents")
plt.title("Count of Dependents")
plt.xlabel("Number of Dependents")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Dependents")
plt.title("Count of Loan Application Outcomes based on Number of Dependents")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Education")
plt.title("Count of Loan Application Outcomes based on Educational Status")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Self_Employed")
plt.title("Count of Loan Application Outcomes based on Self Employed Status")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Credit_History")
plt.title("Count of Loan Application Outcomes based on Credit History")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.countplot(data=df, x="Loan_Status", hue="Property_Area")
plt.title("Count of Loan Application Outcomes based on Property Area")
plt.xlabel("Outcomes")
plt.ylabel("Count")

plt.figure(figsize=(10,7))
sns.scatterplot(data=df, x="LoanAmount", y="ApplicantIncome")
plt.title("Applicant's Income and requested loan amount")
plt.xlabel("Loan Amount")
plt.ylabel("Applicant's Income")

plt.figure(figsize=(10,7))
sns.scatterplot(data=df, x="LoanAmount", y="CoapplicantIncome")
plt.title("Co-Applicant's Income and requested loan amount")
plt.xlabel("Loan Amount")
plt.ylabel("CoApplicant's Income")

plt.figure(figsize=(12,10))
sns.boxplot(data=df, x="Loan_Amount_Term", y="LoanAmount")
plt.title("Requested Loan Amount Term and Loan Amount")
plt.xlabel("Loan Term")
plt.ylabel("Loan Amount")

"""## Feature Engineering

### Encoding the target column
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

if (df['Loan_Status'].dtypes == 'int64'):
  print("Successfully encoded the label/target column")
else:
  print("Failed to encode the target column")

df['Loan_Status'].value_counts()

"""### Filling NaN Values"""

df.isnull().sum()

fill_GenderNan = df['Gender'].mode()[0]
fill_MarriedNan = df['Married'].mode()[0]
fill_DependentNan = df['Dependents'].mode()[0]
fill_SelfEmployedNan = df['Self_Employed'].mode()[0]
fill_LoanAmtNan = round(df['LoanAmount'].mean())
fill_LoanAmtTermNan = df['LoanAmount'].mode()[0]
fill_CreditHistoryNan = df['Credit_History'].mode()[0]

df['Gender'].fillna(fill_GenderNan, inplace=True)

df['Married'].fillna(fill_MarriedNan, inplace=True)
df['Dependents'].fillna(fill_DependentNan, inplace=True)
df['Self_Employed'].fillna(fill_SelfEmployedNan, inplace=True)
df['LoanAmount'].fillna(fill_LoanAmtNan, inplace=True)
df['Loan_Amount_Term'].fillna(fill_LoanAmtTermNan, inplace=True)
df['Credit_History'].fillna(fill_CreditHistoryNan, inplace=True)

NaN_Columns = ['Gender', 'Married', 'Dependents', 'Self_Employed',  'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']

for columns in NaN_Columns:
  print( "Unique values in this column \n"  ,df[columns].value_counts() )
  print('\n')
  print("Number of Null Values",df[columns].isnull().sum() )
  print('\n')
  print("Decription of this columns \n", df[columns].describe() )
  print('\n')
  print("================================================ \n")

"""### Removing Categorical Values"""

df['Dependents'].replace({"3+": "3"}, inplace=True)

df['Dependents'].value_counts()

"""### One-Hot Encoding the Categorical Features"""

df.info()

df.columns

categorical_columns = ['Gender', 'Married', 'Education',
       'Self_Employed', 'Property_Area']

AreYouMale = pd.get_dummies(df['Gender'])
AreYouMarried = pd.get_dummies(df['Married'], prefix='Married', prefix_sep="_")
AreYouGraduated = pd.get_dummies(df['Education'])
AreYouSelfEmployeed = pd.get_dummies(df['Self_Employed'], prefix='SelfEmployment', prefix_sep="_")
WhereDoYouLive = pd.get_dummies(df['Property_Area'])

loan_df = pd.concat([df, AreYouMale, AreYouMarried, AreYouGraduated, AreYouSelfEmployeed, WhereDoYouLive], axis='columns')

loan_df.head()

loan_df.columns

final_df = loan_df[['Male', 'Married_Yes', 'Dependents',
         'Graduate', 'SelfEmployment_Yes', 'ApplicantIncome',
         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
         'Credit_History', 'Rural', 'Semiurban', 'Loan_Status']]

final_df.head()

final_df['Dependents'] = pd.to_numeric(final_df['Dependents'])

final_df.info()

plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(final_df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)

"""## Creating Train, Test and Validation Set"""

X = final_df.drop('Loan_Status', axis='columns')
y = final_df['Loan_Status']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
X_val_scaled = sc.fit_transform(X_val)

print(X_train.shape, X_test.shape, X_val.shape)

print(X_train_scaled.shape, X_test_scaled.shape, X_val_scaled.shape)

"""## Building Machine Learning Models

### Importing Metrics
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""### LazyPredict Method"""

pip install lazypredict

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import RandomizedSearchCV

print("\n\n Lazy Predicts on non-scaled data")
print("===================================== \n")

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models

print("\n\n Lazy Predicts on scaled data")
print("===================================== \n")

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)
models

"""### Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt','log2']
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
min_samples_split = [2, 5, 10,14]
min_samples_leaf = [1, 2, 4,6,8]


rfc_params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}

print(rfc_params)

rfc_RandomizedSearchCV = RandomizedSearchCV(estimator = rfc,
                               param_distributions = rfc_params,
                               n_iter=10, cv=3, verbose=2, n_jobs=-1)

rfc_RandomizedSearchCV.fit(X_train,y_train)

rfc_RandomizedSearchCV.best_estimator_

rfc_RandomizedSearchCV.best_params_

rfc_V2 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=230, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=6, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=1400,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

rfc_V2.fit(X_train,y_train)

y_preds_rfc = rfc_V2.predict(X_test)

print("Accuracy of the Random Forest Model: ", accuracy_score(y_test, y_preds_rfc))

"""### XGBoost Classifier"""

from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb_params = {
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]    
}

print(xgb_params)

xgb_RandomizedSearchCV = RandomizedSearchCV(estimator = xgb,
                               param_distributions = xgb_params,
                               n_iter=10, cv=3, verbose=2, n_jobs=-1)

xgb_RandomizedSearchCV.fit(X_train,y_train)

xgb_RandomizedSearchCV.best_estimator_

xgb_RandomizedSearchCV.best_params_

xgb_V2 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.1,
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

xgb_V2.fit(X_train, y_train)

y_preds_xgb = xgb_V2.predict(X_test)

print("Accuracy of the XGBoost Model: ", accuracy_score(y_test, y_preds_xgb))

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

"""### Logistic Regression"""

logReg_params = [
      {
          'penalty' : ['11', '12', 'elasticnet', 'none'],
          'C' : np.logspace(-4, 4 ,20),
          'solver' : ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
          'max_iter' : [100, 1000, 2500, 5000] 
      }      
]

lr = LogisticRegression()

lr_RandomizedSearchCV = RandomizedSearchCV(estimator=lr, param_distributions= logReg_params, n_iter=10, cv=3, verbose=2,
                               random_state=100, n_jobs=-1)

lr_RandomizedSearchCV.fit(X_train,y_train)

lr_RandomizedSearchCV.best_estimator_

lr_RandomizedSearchCV.best_params_

lr_V2 = LogisticRegression(C=0.0001, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=5000,
                   multi_class='auto', n_jobs=None, penalty='none',
                   random_state=None, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)

lr_V2.fit(X_train_scaled, y_train)

y_preds_lr = lr_V2.predict(X_test_scaled)

print("Accuracy of the Logistic Regression Model: ", accuracy_score(y_test, y_preds_lr))

"""## Evaluating the Models"""

print("Confusion Matrix of the Random Forest Model \n")
print(confusion_matrix(y_test, y_preds_rfc))

print("Confusion Matrix of the XGBoost Model \n")
print(confusion_matrix(y_test, y_preds_xgb))

print("Classification report of the Logistic Regression Model \n")
print(confusion_matrix(y_test, y_preds_lr))

print("Classification report of the Random Forest Model \n")
print(classification_report(y_test, y_preds_rfc))

print("Classification report of the XGBoost Model \n")
print(classification_report(y_test, y_preds_xgb))

print("Classification report of the Logistic Regression Model \n")
print(classification_report(y_test, y_preds_lr))

"""## Cross Validating the Model"""

from sklearn.model_selection import cross_val_score

validation_score_rfc = cross_val_score(rfc_V2, X= X_val, y= y_val, cv=5, scoring= 'accuracy') 
validation_score_xgb = cross_val_score(xgb_V2, X= X_val, y= y_val, cv=5, scoring= 'accuracy') 
validation_score_lr = cross_val_score(lr_V2, X= X_val_scaled, y= y_val, cv=5, scoring= 'accuracy')

print("Validation Score of the Random Forest Model: ")
print(validation_score_rfc)

print("\n")
print("Validation Score of the XGBoost Model: ")
print(validation_score_xgb)

print("\n")
print("Validation Score of the Logistic Regression Model: ")
print(validation_score_lr)

print("Average Validation Score of the Random Forest Model: ")
print(validation_score_rfc.mean())

print("\n")
print("Average Validation Score of the XGBoost Model: ")
print(validation_score_xgb.mean())

print("\n")
print("Average Validation Score of the Logistic Regression Model: ")
print(validation_score_lr.mean())

"""## Saving the Model"""

from joblib import dump

MODEL_NAME = "Loan_Predictor.pkl"

dump(rfc_V2, MODEL_NAME)

