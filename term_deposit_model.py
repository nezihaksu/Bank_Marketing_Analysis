import term_deposit_subscription_eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

SEED = 42

df = pd.read_csv('bank_marketing.csv')
df.drop(df.columns.values[0],axis = 1,inplace = True)
n = df.shape[0]
k = df.shape[1]

categorical_features,df = term_deposit_subscription_eda.convert_categorical(df)
numerical_features = [name for name in df.columns.values if name not in categorical_features]

categorical_df = df[categorical_features]

le = LabelEncoder()
#LabelEncoder,because education is an ordinal categorical variable.
df.education =  le.fit_transform(df.education)
#Non-ordinal categoricals.
other_categoricals = [feature for feature in categorical_features if feature not in ['education','y']]
categorical_df = pd.get_dummies(df[other_categoricals],prefix_sep = '_',drop_first = True)
categorical_df['education'] = df.education
#No-Yes to 0-1.
df.y = pd.get_dummies(df.y)

x = df.drop('y',axis = 1)
categorical_x = categorical_df
numerical_x = df[numerical_features]
y = df['y']

def data_splitter(x,y):
	x_train,x_test_validation,y_train,y_test_validation = train_test_split(x,y,test_size = 0.25,random_state = SEED)
	x_validation,x_test,y_validation,y_test = train_test_split(x_test_validation,y_test_validation,test_size = 0.5,random_state = SEED)
	return x_train,y_train,x_validation,y_validation

categorical_x_train,y_train,categorical_x_validation,y_validation = data_splitter(categorical_x,y)
numerical_x_train,y_train,numerical_x_validation,y_validation =  data_splitter(numerical_x,y)

gnb = GaussianNB()
cnb = CategoricalNB()

class ClfSwitcher(BaseEstimator):

	def __init__(self, estimator = LogisticRegression()):
	    """
	    A Custom BaseEstimator that can switch between classifiers.
	    :param estimator: sklearn object - The classifier
	    """ 
	    self.estimator = estimator
	
	def fit(self, X, y=None, **kwargs):
	    self.estimator.fit(X, y)
	    return self
	
	def predict(self, X, y=None):
	    return self.estimator.predict(X)
	
	def predict_proba(self, X):
	    return self.estimator.predict_proba(X)
	
	def score(self, X, y):
	    return self.estimator.score(X, y)

gnb_parameters = {
	'clf__estimator':[gnb],
	'clf__estimator__var_smoothing':[1*10**(-9),1*10**(-10)]
}

cnb_parameters = {
	'clf__estimator':[cnb],
	'clf__estimator__alpha':[1.0,0.9,1.1],
	'clf__estimator__fit_prior':[True,False]
}

pipeline = Pipeline([
    ('clf', ClfSwitcher()),
])


#Function to get probability results of categorical and numerical data and multiply them to get the final outcome.
def bayesian_model(clf_parameters,x_train,y_train,x_validation):

	grid_search = GridSearchCV(pipeline,clf_parameters,n_jobs=-1, verbose=1,cv = 5)	
	grid_search.fit(x_train,y_train)
	prob = grid_search.predict_proba(x_validation)

	return prob
	
categorical_bayesian_prob = bayesian_model(cnb_parameters,categorical_x_train,y_train,categorical_x_validation)
numerical_bayesian_prob = bayesian_model(gnb_parameters,numerical_x_train,y_train,numerical_x_validation)
bayesian_score = categorical_bayesian_prob*numerical_bayesian_prob

print('Bayesian Score is:',bayesian_score)

x_train,y_train,x_validation,y_validation = data_splitter(x,y)

logreg = LogisticRegression()
lgbm = LGBMClassifier()

log_parameters = {
	'clf__estimator':[logreg]
    'clf__estimator__penalty': ['l1','l2'],
    'clf__estimator__max_iter': [4000],
    'clf__estimator__solver':['saga']
   }
lgbm_parameters = {
	'clf__estimator':[lgbm]
    'clf__estimator__boosting_type' : ['gbdt','dart'],
    'clf__estimator__num_leaves': ['31','20'],
    'clf__estimator__learning_rate': [0.1,0.01],
    'clf__estimator__n_estimators':[100,50]
   }

def model_selection(clf_parameters,x_train,y_train,x_validation,y_validation):

	grid_search = GridSearchCV(pipeline,clf_parameters,n_jobs=-1, verbose=1,cv = 5)
	grid_search.fit(x_train,y_train)
	
	return grid_search.best_estimator_,grid_search.best_score_,grid_search.score(x_validation,y_validation)

log_best_estimators,log_best_score,log_score = model_selection(log_parameters,x_train,y_train,x_validation,y_validation)
lgbm_best_estimators,lgbm_best_score,lgbm_score = model_selection(lgbm_parameters,x_train,y_train,x_validation,y_validation)

print('Log Best estimators:',log_best_estimators)
print('Log Best Score:',log_best_score)
print('Log Validation Score:',log_score)
print('LGBM Best Estimators:',lgbm_best_estimators)
print('LGBM Best Score:',lgbm_best_score)
print('LGBM Validation Score:',lgbm_score)