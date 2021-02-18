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

#Duration effects target variable pretty heavily,because if duration = 0,then there is no call made,therefore y=0,no subscription has been made.
#Due to that duration is not appropriate for a real life model.
#Treating education as a non-ordinal categorical features yields better prediction power.Meaning,there is not relation between education levels.
categorical_features = ['job', 'marital', 'education','day','age','default', 'housing', 'loan', 'contact','month', 'poutcome']
numeric_features = ['balance','campaign','pdays', 'previous']
print("Numeric Features:\n")
print(len(numeric_features))
print("Categorical features:\n")
len(categorical_features)


#FEATURE ENGINEERING

#Converting age into categorical data by putting values for certain age ranges.
#This will make model to treat people according to age range(young,mature,old) as we society does in real life.
print(pre_df.age.describe())

pre_df.age[(pre_df.age > 18) & (pre_df.age <=25)] = 1
pre_df.age[(pre_df.age > 25) & (pre_df.age <=65)] = 2
pre_df.age[(pre_df.age > 65) & (pre_df.age <=95)] = 3
print(pre_df.age)

le = LabelEncoder()
#LabelEncoder,because education is an ordinal categorical variable.
#Thus there will be contribution hierarchy among categories. 
pre_df.education =  le.fit_transform(pre_df.education)
print(pre_df.education)
pre_df.y = le.fit_transform(pre_df.y)
print(pre_df.y)

#Due to all categorical features are being non-ordinal OneHotEncoding is applied by using get_dummies.
dummy_df = pd.get_dummies(pre_df[categorical_features],prefix_sep = '_',drop_first = True,columns = categorical_features)
print("OneHotEncoded nominal categorical features.")
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

#Balancing the target percentage.Data augmentation for the minority class.
smote = SMOTE(sampling_strategy = 'minority')
balanced_x,balanced_y = smote.fit_sample(x,y)
print("======= Balanced Dataset ========")
print(pd.DataFrame(balanced_x,columns=[x.columns]))


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


lgbm = lightgbm.LGBMClassifier(learning_rate = 0.1,
                               min_child_samples = 20,
                               min_split_gain=1,
                               min_child_weight = 0.1,
                               reg_alpha = 0.01,
                               reg_lambda = 0.01,
                               objective = "binary",
                               importance_type = "gain",
                               random_state = SEED)
lgbm.fit(balanced_x_train,balanced_y_train)

scores = cross_val_score(lgbm,balanced_x_test,balanced_y_test,cv = 10) 
acc_df = pd.DataFrame(data = {"grad_boosted_trees": range(10),"accuracy": scores})
acc_df.plot(x="grad_boosted_trees",y="accuracy",marker="o",linestyle="--")
print("Mean of 10 scores is "+str(np.mean(scores)))

plot_confusion_matrix(lgbm,balanced_x_test,balanced_y_test,display_labels = ["Subscribed","Not Sub."])

#ROC curve visualization shows the classifier's ability to classify positive and negative labels at each threshold.
#AUC is better metric than accuracy score and error rate when it comes to the Classifiers.
#The bigger the area under the curve the better the at classifying both positive and negative labels.
#It also depends on our research topic,if classifying positive labels are more important than the negative ones
#it enables us to see which threshold to use to achieve that purpose.
plot_roc_curve(lgbm,balanced_x_test,balanced_y_test)

#Using SVM for classifying.
#Picking dataset as balanced for SVM classifier.

#Calculation of kernel makes SVM take lots of time to make to grid search over while fitting the dataset.
#Because kernel looks for appropriate dimension where it can separate labels.
#Due to that reason,instead of generating data to make it balanced,it rathered to skim the data to make them equal numbers.
#And still we have about more than 10,500 instances to fit the model into.
svm_df = pd.concat([pre_df[pre_df.y == 1],pre_df[pre_df.y == 0][:num_pos_label]],axis = 0)

#Also i shuffled the data randomly.
svm_df = shuffle(svm_df,random_state = 42).reset_index()
svm_df.drop("index",axis = 1,inplace = True)
print("SVM Dataset")
print(svm_df)

svm_dummy_df = pd.get_dummies(svm_df[categorical_features],prefix_sep = '_',drop_first = True,columns = categorical_features)
print(svm_dummy_df)

svm_model_df = pd.concat([svm_df[numeric_features],svm_dummy_df,svm_df.y],axis=1)
print(svm_model_df)

svm_x = svm_model_df.iloc[:,:-1]
svm_y = svm_model_df.y
print("==== Target ====")
print(svm_y)
print("==== Features ====")
print(svm_x)

x_train,x_test,y_train,y_test = train_test_split(svm_x,svm_y,test_size = 0.25,random_state = SEED)
x_train_scaled = scale(x_train)
x_test_scaled = scale(x_test)

clf_dummy = DummyClassifier(strategy="most_frequent",random_state = SEED)
clf_dummy.fit(x_train,y_train)
print("Dummy classifier Score is: " + str(clf_dummy.score(x_test,y_test)))

#Grid search over model to find best hyperparameters to fit the model to the data.
print("Grid search over the SVM...")
svm = SVC(random_state = SEED)
param_grid = [{"C":[0.5,1,10,100],
               "gamma":["scale",1,0.1,0.01,0.001,0.0001],
               "kernel":["rbf"],},]
optimal_params = GridSearchCV(SVC(),
                              param_grid,
                              cv=3,
                              scoring="accuracy")
optimal_params.fit(x_train_scaled,y_train)

best_params = optimal_params.best_params_

print("Best hyperparameters: " + str(best_params))


#Using best hyperparameters to fit the model.
clf_svm = SVC(C = best_params["C"],gamma = best_params["gamma"],kernel = best_params["kernel"],random_state = 42)
clf_svm.fit(x_train_scaled,y_train)
plot_confusion_matrix(clf_svm,
                      x_test_scaled,
                      y_test,
                      values_format = "d",
                      display_labels = ["Did not Subscribed","Subscribed"])

scores = cross_val_score(clf_svm,x_test_scaled,y_test,cv = 10) 
acc_df = pd.DataFrame(data = {"Machine": range(10),"accuracy": scores})
acc_df.plot(x="Machine",y="accuracy",marker="o",linestyle="--")
print("Mean of 10 scores is "+str(np.mean(scores)))

plot_roc_curve(clf_svm,x_test_scaled,y_test)

#To see the most important features from the tree,used XGBoost classifier.
print("Hyperparameter search for XGBoost classifier...")
params_grid = {
    "max_depth":[3,4,5],
    "learn_rate":[0.1,0.01,0.05],
    "gamma":[0,0.25,1.0],
    "reg_lambda":[0,1.0,10.0]  
}

optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective="binary:logistic",seed=42),
                              param_grid=params_grid,
                              scoring="roc_auc",
                              cv=3
                              )
optimal_params.fit(x_train,y_train,
                   early_stopping_rounds=10,
                   eval_metric="auc",
                   eval_set=[(x_test,y_test)],
                   verbose=True)

best_params = optimal_params.best_params_ 

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective="binary:logistic",
                            gamma=best_params["gamma"],
                            learn_rate=best_params["learn_rate"],
                            max_depth=best_params["max_depth"],
                            reg_lambda=best_params["reg_lambda"]
                            )
clf_xgb.fit(x_train,y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric="aucpr",
            eval_set=[(x_test,y_test)])


plot_roc_curve(clf_xgb,x_test,y_test)
plt.show()

bst = clf_xgb.get_booster()
for importance_type in ("weight","gain","cover","total_gain","total_cover"):
  print("{} ".format(importance_type),bst.get_score(importance_type=importance_type))

node_params = {"shape":"box",
               "style":"filled,rounded",
               "fillcolor":"#78cbe"}
leaf_params = {"shape":"box",
               "style":"filled",
               "fillcolor":"#e48038"}
xgb.to_graphviz(clf_xgb,num_trees=0,size="20,15",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)
plt.show()