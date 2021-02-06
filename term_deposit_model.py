# import term_deposit_subscription_eda
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB,CategoricalNB
# from sklearn.preprocessing import LabelEncoder
# from sklearn.base import BaseEstimator
# from sklearn.pipeline import Pipeline

# SEED = 42

# df = pd.read_csv('bank_marketing.csv')
# df.drop(df.columns.values[0],axis = 1,inplace = True)
# n = df.shape[0]
# k = df.shape[1]


# class Preprocess(object):
	
# 	categorical_features,df = term_deposit_subscription_eda.convert_categorical(df)
# 	numerical_features = [name for name in df.columns.values if name not in categorical_features]
# 	categorical_df = df[categorical_features]


# le = LabelEncoder()
# #LabelEncoder,because education is an ordinal categorical variable.
# df.education =  le.fit_transform(df.education)
# #Non-ordinal categoricals.
# other_categoricals = [feature for feature in categorical_features if feature not in ['education','y']]
# categorical_df = pd.get_dummies(df[other_categoricals],prefix_sep = '_',drop_first = True)
# categorical_df['education'] = df.education
# #No-Yes to 0-1.
# df.y = pd.get_dummies(df.y)

# x = df.drop('y',axis = 1)
# categorical_x = categorical_df
# numerical_x = df[numerical_features]
# y = df['y']

# def data_splitter(x,y):
# 	x_train,x_test_validation,y_train,y_test_validation = train_test_split(x,y,test_size = 0.25,random_state = SEED)
# 	x_validation,x_test,y_validation,y_test = train_test_split(x_test_validation,y_test_validation,test_size = 0.5,random_state = SEED)
# 	return x_train,y_train,x_validation,y_validation

# categorical_x_train,y_train,categorical_x_validation,y_validation = data_splitter(categorical_x,y)
# numerical_x_train,y_train,numerical_x_validation,y_validation =  data_splitter(numerical_x,y)

# gnb = GaussianNB()
# cnb = CategoricalNB()

# class ClfSwitcher(BaseEstimator):

# 	def __init__(self, estimator = LogisticRegression()):
# 	    """
# 	    A Custom BaseEstimator that can switch between classifiers.
# 	    :param estimator: sklearn object - The classifier
# 	    """ 
# 	    self.estimator = estimator
	
# 	def fit(self, X, y=None, **kwargs):
# 	    self.estimator.fit(X, y)
# 	    return self
	
# 	def predict(self, X, y=None):
# 	    return self.estimator.predict(X)
	
# 	def predict_proba(self, X):
# 	    return self.estimator.predict_proba(X)
	
# 	def score(self, X, y):
# 	    return self.estimator.score(X, y)

# gnb_parameters = {
# 	'clf__estimator':[gnb],
# 	'clf__estimator__var_smoothing':[1*10**(-9),1*10**(-10)]
# }

# cnb_parameters = {
# 	'clf__estimator':[cnb],
# 	'clf__estimator__alpha':[1.0,0.9,1.1],
# 	'clf__estimator__fit_prior':[True,False]
# }

# pipeline = Pipeline([
#     ('clf', ClfSwitcher()),
# ])


# #Function to get probability results of categorical and numerical data and multiply them to get the final outcome.
# def bayesian_model(clf_parameters,x_train,y_train,x_validation):

# 	grid_search = GridSearchCV(pipeline,clf_parameters,n_jobs=-1, verbose=1,cv = 5)	
# 	grid_search.fit(x_train,y_train)
# 	prob = grid_search.predict_proba(x_validation)

# 	return prob

# categorical_bayesian_prob = bayesian_model(cnb_parameters,categorical_x_train,y_train,categorical_x_validation)
# numerical_bayesian_prob = bayesian_model(gnb_parameters,numerical_x_train,y_train,numerical_x_validation)
# bayesian_score = categorical_bayesian_prob*numerical_bayesian_prob

# print('Bayesian Score is:',bayesian_score)

# x_train,y_train,x_validation,y_validation = data_splitter(x,y)

# logreg = LogisticRegression()
# lgbm = LGBMClassifier()

# log_parameters = {
# 	'clf__estimator':[logreg],
#     'clf__estimator__penalty': ['l1','l2'],
#     'clf__estimator__max_iter': [4000],
#     'clf__estimator__solver':['saga']
#    }
# lgbm_parameters = {
# 	'clf__estimator':[lgbm],
#     'clf__estimator__boosting_type' : ['gbdt','dart'],
#     'clf__estimator__num_leaves': ['31','20'],
#     'clf__estimator__learning_rate': [0.1,0.01],
#     'clf__estimator__n_estimators':[100,50]
#    }

# def model_selection(clf_parameters,x_train,y_train,x_validation,y_validation):

# 	grid_search = GridSearchCV(pipeline,clf_parameters,n_jobs=-1, verbose=1,cv = 5)
# 	grid_search.fit(x_train,y_train)
	
# 	return grid_search.best_estimator_,grid_search.best_score_,grid_search.score(x_validation,y_validation)

# log_best_estimators,log_best_score,log_score = model_selection(log_parameters,x_train,y_train,x_validation,y_validation)
# lgbm_best_estimators,lgbm_best_score,lgbm_score = model_selection(lgbm_parameters,x_train,y_train,x_validation,y_validation)

# print('Log Best estimators:',log_best_estimators)
# print('Log Best Score:',log_best_score)
# print('Log Validation Score:',log_score)
# print('LGBM Best Estimators:',lgbm_best_estimators)
# print('LGBM Best Score:',lgbm_best_score)
# print('LGBM Validation Score:',lgbm_score)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import lightgbm
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
SEED = 42

pre_df = pd.read_csv("C:\Users\nezih\Desktop\data\bank\cleaned_bank_dataset.csv")
print(pre_df)
#Data types.
print(pre_df.dtypes)

def convert_categorical(df):
    feature_list = df.columns
    categorical_features = []
    letter_pattern = re.compile(r'[A-z]')
    #If values types are all str or int,it is impossible to distinguish them with this method,so i prefer to do it with regex.
    for column in feature_list:

        if letter_pattern.match(str(df[column].values[0])):
            df[column] = pd.Categorical(df[column])
            categorical_features.append(df[column].name)
        
    return list(set(categorical_features))

print(convert_categorical(pre_df))
print(pre_df.dtypes)

categorical_features = pre_df.dtypes[pre_df.dtypes == "category"].index
numeric_features = [feature for feature in pre_df.columns if feature not in categorical_features]
print("Numeric Features:\n")
print(numeric_features)
print("Categorical features:\n")
categorical_features

le = LabelEncoder()
#LabelEncoder,because education is an ordinal categorical variable.
#Thus there will be contribution hierarchy among categories. 
pre_df.education =  le.fit_transform(pre_df.education)
print(pre_df.education)
pre_df.y = le.fit_transform(pre_df.y)
print(pre_df.y)

non_ordinal_categoricals = [feature for feature in categorical_features if feature not in ['education',"y"]]
dummy_df = pd.get_dummies(pre_df[non_ordinal_categoricals],prefix_sep = '_',drop_first = True)
print("Dataset after getting dummies.")
print(dummy_df)

#Numeric feature number is 7,dummy_df column size is 32 plus education and y variable makes 41 columns.
model_df = pd.concat([pre_df[numeric_features],dummy_df,pre_df.education,pre_df.y],axis=1)
print("Dataset before modelling")
print(model_df)

#When y column dropped,rest of the dataframe converts into 45210 rows for some reason,so in this way it is prevented.
x = model_df.iloc[:,:-1]
print("X columns")
print(x)

y = model_df.y
print("Y column")
print(y)

#There are 7.5 times more negative labels than there are positives.This causes imbalance.
#If used as a null classifier,a new sample will be labeled as negative 7.5 times more than positive.
label_num_ratio = len(model_df.y[model_df.y == 0])/len(model_df.y[model_df.y == 1])
print("Model imbalance between the labels")
print(label_num_ratio)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = SEED)

def imbalance_test(x_train,x_test,y_train,y_test,generated_prediction):
  clf_dummy = DummyClassifier(strategy="constant",random_state = SEED,constant=generated_prediction)
  clf_dummy.fit(x_train,y_train)
  return clf_dummy.score(x_test,y_test)
#Dummy classifier generates 0 as prediction and yields 0.88 score.
print("Negative prediction score:")
print(imbalance_test(x_train,x_test,y_train,y_test,0))
#Dummy classifier generates 1 as prediction and yields 0.12 score.
print("Positive prediction score")
print(imbalance_test(x_train,x_test,y_train,y_test,1))
#This goes to show how imbalance effects predictions.

#Balancing the target percentage.
smote = SMOTE(sampling_strategy = 'minority')
x,balanced_y = smote.fit_sample(x,y)
sc = MinMaxScaler()
balanced_x = sc.fit_transform(x)

print("X shape after SMOTE generation.")
print(balanced_x.shape)
print("Y shape after SMOTE generation.")
print(balanced_y.shape)
num_generated_instances = len(balanced_x)-len(model_df)
print("Number of generated instances to balance the labels:")
print(num_generated_instances)

balanced_x_train,balanced_x_test,balanced_y_train,balanced_y_test = train_test_split(balanced_x,balanced_y,test_size = 0.25,random_state = SEED)

#Now the each label prediction is close to each other as much as possible.
print("Negative prediction score after SMOTE:")
print(imbalance_test(balanced_x_train,balanced_x_test,balanced_y_train,balanced_y_test,0))
print("Positive prediction score after SMOTE:")
print(imbalance_test(balanced_x_train,balanced_x_test,balanced_y_train,balanced_y_test,1))

#Using logistic regression to see how much difference there is between cross validations in terms of model score to detect if dataset is noisy.
lr = LogisticRegression(max_iter=4000,random_state=SEED)
scores = cross_val_score(lr,balanced_x_train,balanced_y_train,cv = 10)
acc_df = pd.DataFrame(data = {"log_model_num": range(10),"accuracy": scores})
acc_df.plot(x="log_model_num",y="accuracy",marker="o",linestyle="--")
plt.show()

plot_confusion_matrix(lr,balanced_x_test,balanced_y_test,display_labels = ["Subscribed","Not Sub."])
plt.show()