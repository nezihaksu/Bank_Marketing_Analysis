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