# Term Deposit Subscription Dataset Analysis

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

## Exploratory Data Analysis

### Balance

I wrote a function that takes arguments to group by the dataframe by a numerical feature and the kind of it (sum or mean).
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
More time on the phone spent on people who did not accept to subscribe than the ones who subscribed.Which goes to show lots of time spent on persuasion.

![](/graph_images/sum_of_duration1.JPG)
![](/graph_images/sum_of_duration2.JPG)





