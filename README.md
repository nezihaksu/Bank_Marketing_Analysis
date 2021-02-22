# Term Deposit Subscription Dataset Analysis

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

## Exploratory Data Analysis

### Balance

Balance accumucalation is between the ages 30 and 60's.Term deposit subscription is also highly correlated with age and balance.

The bank has been targeting the people who are between the ages 30-60 yet the number of negative instances in dataset are overwhelmingly high compared to positives.

![](/graph_images/balance_age_scatter.JPG)

I wrote a function that takes arguments to group by the dataframe by a numerical feature and the kind of it (sum or mean).

```python
def groupby_method(groupby_features,method):
        series_groupby = df[groupby_features].groupby(df[groupby_features].iloc[:,-1].name)
        if method == 'sum':
            return series_groupby.sum()
        elif method == 'mean':
            return series_groupby.mean()
```

Firstly,to see the graphs where categorical features is grouped by Balance.Categorical features shows the mean balance of people in those respective categories.

It shows how much balance does most of the people in each category have.

According to subscription and balance graph,people who did not subscribed has more balance than who did.

Management type of jobs also have the most of balance amount whereas unkown category has the least amount.

Balance difference between secondary and tertiary education is so small.

People who have house loan have more balance.


![](/graph_images/balance1.JPG)
![](/graph_images/balance2.JPG)


### Duration

I used the same function to see the time spent on phone to make people subscribe the term deposit.

Sum of durations of phone calls for each category.

Longest phone calls happened with blue collar,married people.

Also people who use cellphone talked more.

Lots of time also spent on people who we don't know whether or not subscribed.

More time on the phone spent on people who did not accept to subscribe than the ones who subscribed.This again shows how imbalanced the dataset is,since most of the contacts was not succesful.


![](/graph_images/sum_of_duration1.JPG)
![](/graph_images/sum_of_duration2.JPG)

#### Note: Duration can't be added to a model if we want it to be as realistic as possible,because when there is no call,duration and term deposit variables are 0.And this means we automatically assume person who has not been contacted would not subscribe the term deposit,which is a unsupported claim.

### Categorical Feature Distribution

This function i wrote enables me to see the distribution of categories in each categorical features,thus helps me to see prevelant categories and relate them with the context.

```python
#Creating a category dict where each category name is key and their count are the value.
def category_distribution(df,categorical_features):
  feature_dict = {}
  for category in categorical_features:
    category_dict = {}
    for unique_data in df[category].unique():
      category_dict[unique_data] = len(df[df[category] == unique_data])
      feature_dict[category] = category_dict
```

![](/graph_images/categorical_feature_distribution1.JPG)
![](/graph_images/categorical_feature_distribution2.JPG)
![](/graph_images/categorical_feature_distribution3.JPG)

## Model

### Dummy Classifier

First i looked at the features if any of them any modification.

Converted categorical features into numerical representation with pandas' get_dummies method.

After that i checked if converting age into categorical variable by assigning certain range of ages as "young","mature","old",which improved nothing.

For a comparison with the models i needed a null accuracy of sorts,because of this need i decided to use sklearn's DummyClasifier.
```python
def imbalance_test(x_train,x_test,y_train,y_test,generated_prediction):
  clf_dummy = DummyClassifier(strategy="constant",random_state = SEED,constant=generated_prediction)
  clf_dummy.fit(x_train,y_train)
  return clf_dummy.score(x_test,y_test)
```
I chose constant strategy to see each class' accuracy power and found these results:

	Negative prediction score:
	0.8802972662125099
	Positive prediction score
	0.11970273378749005

### Gradient Boost and Upsampling

Due to dataset being extremely imbalanced,i decided to fit Gradient Boosting model with SMOTE.

Boosting process helps each class to be represented in the model better rather than bagging and maximum likelihood,even though boosting model has previous prediction in its calculations,it tests each of its tree to yield lowest residuals in every iteration,thus reducing the minority class' residual.

SMOTE generates instances to make classes equal.It is a very reliable libriary.It generates instances in certain ranges that they appear in the dataset.

Null accuracy of each class after SMOTE:

	Negative prediction score:
	0.49842192274936126
	Positive prediction score
	0.5015780772506387

Gradient Boost model after a grid search over its hyperparameters with cross validation:

```pyhon
lgbm = lightgbm.LGBMClassifier(learning_rate = 0.1,
                               min_child_samples = 20,
                               min_split_gain=1,
                               min_child_weight = 0.1,
                               reg_alpha = 0.01,
                               reg_lambda = 0.01,
                               objective = "binary",
                               importance_type = "gain",
                               random_state = SEED)
```

### Note: When it comes to performance metric of a classifier,ROC Curve is better than accuracy score and error rate calculated from Confusion Matrix.Since it shows model's ability to detect true positives and false positives at every possible threshold between the classes.This also helps to adjust the model's threshold for our needs when it is more important to detect a class than the other class.For more information please check [this paper](https://www.researchgate.net/publication/2364670_The_Effect_of_Class_Distribution_on_Classifier_Learning_An_Empirical_Study).

![](/graph_images/lgbm_upscaled_data_roc_curve.JPG)

### Support Vector Machine and Downsampling

After this step i wanted to approach the imbalance problem in a reverse way by downsampling instead of upsampling.

Downsampling causes loss of information.However it is best tecnique deployed against imbalance problem in terms of time spent on dataset.

To use downsampling for my advantage i wanted to use downsampled data with Support Vector Machine,since it takes lots of time to train it with big dataset because of kernel calculations.

Support Vector Machine finds the hyperplane that separates two classes in infinite dimensions with Kernel Trick.

SVM is really great separating overlapping data,a data that requires non-linear separation and high dimensional data. 

After a grid search for hyperparameters:

```python
clf_svm = SVC(C = best_params["C"],gamma = best_params["gamma"],kernel = best_params["kernel"],random_state = SEED)
clf_svm.fit(x_train_scaled,y_train)
```

Roc curve of SVM classsifier:

![](/graph_images/svm_downscaled_data_roc_curve.JPG)

Support vector classifies positives and negatives nearly perfectly.Greater the AUC better the classifying.


### Extreme Gradient Boost and Downsampling

To catch the accuracy of Support Vector Machine and explainability of Gradient Boost Trees using XGBoost is a good choice.

XGBoost have lots of optimization factors and designed to solve any kind of machine learning problem.

It also depends on the prior probability and because of that sensitive to imbalances.

Robust to outlier in the dataset.

```python
clf_xgb = xgb.XGBClassifier(seed=42,
                            objective="binary:logistic",
                            gamma=best_params["gamma"],
                            learn_rate=best_params["learn_rate"],
                            max_depth=best_params["max_depth"],
                            reg_lambda=best_params["reg_lambda"]
                            )
```

ROC Curve to see how well does it classifies both positive and negative samples:

![](/graph_images/xgboost_downsampled_roc.JPG)

It has the same performance as the SVM.

## In conclusion:

When dealing with a imbalance dataset,the dataset definitely needs to be modified either by upsampled or downsampled otherwise the model would be inclined to detect more of the majority class.

Using downsampled version of the data with more powerful models yielded better outcomes and ables us to train faster.

Gradient Boost and XGBoost have advantage of explainability over SVM model.

Gradient Boost feature importance shows that having housing debt,the month and the day of the contacts are the most important factor when it comes to the subscription.

```python
feature_importances = zip(lgbm.feature_importances_,x.columns)
sorted_importances = sorted(feature_importances,reverse=True)
feature_names = []
importance_values = []
for k,v in sorted_importances:
  feature_names.append(v)
  importance_values.append(k)
plt.bar(feature_names[:4],importance_values[:4])
```
![](/graph_images/lgbm_feature_importance.JPG)

XGBoost also rooted its tree with the features that have the highest similiarity scores.

According to it,it is also the case that most important factors are unknown contacts and time related features.Especially the month may.

![](/graph_images/xgboost_tree.JPG)

These results can be attributed to the planning of the bank,when they call people mostly in the month may.Also contacting with people that they don't know what the contact device is.

It shows that these models are focused on the repetation of categorical features.

The more the category repeats in the dataset the more it has the chance of effecting the models classification power.

Bank is contacting people in month may since it is the time when they decide to buy a new home,therefore housing debt is also a big factor.

By calling people (in the month may) who has housing debt or at a time of their lives where they need to buy a house bank can have a more detailed and balanced dataset,since it would yield more positive outcomes.

And it would also result in less contact,less phone bill,less salary paid to workers.
