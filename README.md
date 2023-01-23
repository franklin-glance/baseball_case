# Baseball Case 
See `baseball_writeup_withcode.ipynb` for the code along with this writeup. 

`baseball_script.py` contains the code for the drafting algorithm.

## Data Exploration

We are given two datasets. The first dataset contains information about the 50 available players for the draft, and the second dataset contains information about 30 teams in the MLB. 

#### Team Dataset

Given the target of "win more games next season", it is evident that the number of wins of each team is our target variable. 
We must determine which features are most important in predicting the number of wins of a team, and then use those features to identify which players to draft to most improve the performance of our team (team 26, Seattle). 

First, we will look at the distribution of the target variable, W (number of wins). 

The number of wins is relatively normally distributed, with a mean of 80.933 and a standard deviation of 11.085. Our team has 61 wins which is significantly below the mean, which make sense since it is the 26th ranked team out of 30. 

Next, we will look at the relationship between the feature columns and the target variable. 

We can ignore columns that are not present in the player dataset, since those columns cannot be improved by drafting players, and any correlation between those and columns that are effective will be improved by drafting a player that improves the other feature column. 

We can also ignore W and L, since those are inherently directly correlated with the target variable.

Plotting the remainder of the columns, it is evident that some columns are much more correlated than others. 

In ranking our players, we will want to determine a correllation threshold between the features and the target variable. We will use a threshold of 0.25, which is subject to change. 


A threshold of 0.2 results in 8 features being selected, in order of significance: 
- R_hitting: Runs scored 
- RBI: Runs batted in 
- SLG: Slugging percentage
- OBP: On base percentage
- HR: Home runs  
- BA: Batting average
- H: Hits


After selecting these 8 features, we will look at the correlation between each of these features. 

We can see that there are some features that are highly correlated with each other.
This is not a problem, since we are not using these features to predict the target variable, but rather to identify which players to draft. 

#### Player Dataset

##### Exploring the data
The player dataset contains information about 50 players, and we will be using the 8 features selected in the previous section to determine which players to draft, along with the player's cost. 

We will first look at the correlation between the selected features. 

The selected features are somewhat correlated. This is not a problem, since we are not using these features to predict the target variable, but rather to identify which players to draft.

Next, we will look at the correlation between the selected features and the cost of the player.

Interestingly, the 'HR', 'RBI', and 'SLG' are not correlated at all with the cost of the player, even though the correlation between these features and the target variable is high. This suggests that we shouldn't weight the cost of the player too heavily when drafting players, rather we should primarily focus on improving the target metrics of our team while simply focusing on keeping the cost below budget. 


##### Drafting players

In summary, we use several weights in ranking player data, in order of importance:
1. Correlation between features and number of wins in team dataset
2. Delta between our team's current features and the number 1 ranked team's features
3. Budget constraint

Our algorithm for choosing players works in the following manner:

1. Determine signifcant features based on correlation between team data and number of wins. (i.e. pick a correlation threshold, in this case 0.25). 
2. Get our team's current stats from the team dataframe. This will be updated as we draft players. 
3. Scale the player data between 0 and 1. This will be used to ensure that the weights of the features are not biased by metrics with larger values. 
4. Draft:
   1. Each draft round, a new delta value is calculated between our team's current stats and the number 1 ranked team's stats. This is a secondary weighting system beyond the correlation between features and number of wins.
   2. For each available player, a "player score" is calculated. This score is the sum of the product of the weights (correlation and team delta) and the player's scaled feature values. 
   3. The player's player score and budget score are added together to get the total score. The budget score has little impact on the total score unless a player is over budget, in which case the player is not considered.
5. The player's are ranked by their total score, and the player with the highest score is drafted. The current team stats are updated to reflect the drafted player's stats.


