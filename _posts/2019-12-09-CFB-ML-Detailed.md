layout: page
title: "CFB ML Detailed report"
date: 2019-12-08

GridIron: It's Just a Numbers Game
Scalable Machine Learning: Project Report
Team Members: Nathaniel Wiatrek, Mateusz Dyl, Benjamin Musil
12/13/19
Abstract
A number of machine learning models, Linear Regression, GradientBoost, XGBoost, and MultiLayer Perceptrons, predicted the result of an American NCAA Division I football game based on the position of the team combined with statistics about the offense, defense and special teams. The model first generated drives for both teams and then gave a confidence score to predict which team wins at the end of each quarter. To do this, a binary classifier with XGBoost determined if an offensive play would be pass or run. Next, a linear regression model determined the result of that offensive play. These two steps were repeated until either a fourth down was reached or there was a touchdown scored. If there was a fourth down, another model determined the play and result. When each drive ended, the score was updated if necessary and then passed through the end of quarter winner predictor model to determine the result of the game.


Objective
Predicting the winner of a football game, whether it be collegiate or professional, is a large part of the gambling industry. Whenever a football game is listed in the papers or online, there’s usually a “Vegas line” that describes not only what team is predicted to win, but also how many total points there will be in the game and the score differential between the two teams. These odds are predicted by using statistics and analyst assumptions about teams, but they have the potential to be more accurate through machine learning.

For those unfamiliar with American football, two teams take turns having possession of the football (these turns referring to “drives”). The drive is a string of offensive start and stop plays beginning when a team has possession of the ball and ending when they either score, give the ball to the other team (a turnover, e.g. a fumble or interception), or make a defensive play to punt the ball to the other team and force them to start their own drive from a long field position.

Our goal is to create winner predictions of a NCAA Division I football game by giving a confidence prediction based on the teams and the quarter the game is in, while generating offensive drives for each team. We took play by play game data collected by ESPN for NCAA Division I football which is divided into seasons and each week of that season (https://drive.google.com/open?id=0B13YvT1olZIfZ0NKaEZDdGt3TDQ). We combined this data set with offensive and defensive team average statistics for the same season provided by Sports Reference (https://www.sports-reference.com).

The code for this project can be found at https://github.com/nwiatrek/cfb-ml.
Architecture
Predicting the winner of a college football game is a complex process that cannot be easily done with one model. As such the team separated out many decisions/results that a team would have to predict the winner. The architecture is as follows with the drive generator seen in Figure 1.


Figure 1. Drive Prediction FlowChart
Determine what the play will be (pass or run)
Get the result of the play (yardsGained)
Generate drive (does it score)
Predict winner by end of quarter
Predict Play
The Predict play model is a binary classification problem that was solved using XGBoost Classification. The problem is to determine what plays the offense will run: a running play (rush) or a passing play (pass). The following features were given to the model:

'Quarter', 'down', 'distance', 'yardLine', 'Passing Completion', 'Pass Attempts', ‘completion percentage', 'passing yards', 'Passing Touchdowns', 'Rush Attempts', 'Rushing yards', 'rush average', 'Rush Touchdowns', 'First down by pass', 'First down by rush'

These features came from a combination of the ESPN college football play by play data and the Sport Reference team statistics set. They were joined together on the offensive team school name. Distance, yard line, and down of the play were some of the most important features as seen in Figure 2.


Figure 2. Feature importance in the combined data set of game plays and team stats

These features make intuitive sense to football viewers as distance, where the ball is at the start of the play, and play down are heavily put into consideration by football coaches. Another promising factor of the feature importance is that the Pass Attempts and Rush Attempts are similar in weight, meaning that the model will weigh the number of each attempt equally and not scale one highly over the other.

This model was able to achieve a 0.61 AUC on the validation set with an accuracy of  0.62. The  confusion matrix on the validation set is seen in Figure 3.


Figure 3. Confusion matrix to predict a rush or pass as a play

The model is good at determining that the play was a pass when it truly was a pass but struggled determining a rushing play. This model beat out the Multi Layer Perceptron and the gradient boosting classifier in testing.
Rushing Result
The rushing result model was utilized when the play was determined to be a running play. This model is GradientBoostingRegressor. This model was particularly tricky to train as the majority of running plays result in 2-5 yards. To compensate for this, the amount of explosive plays (above 35 yards, or negative yards) was tripled, and the amount of running plays that have zero yards gained was doubled. The resulting model was better able to predict when bigger or worse running plays were happening and the resulting feature importance is in Figure 4.


Figure 4. Feature importance predicting a rushing result

The number one feature for the rushing result was the yard line that the play started on. The next two most important features were the rushing average for the offense as well as the defense’s number of allowed rushing touchdowns.


Figure 5. Statistics of the train and test data frames to predict a rush play
As Figure 5 shows, this feature engineering allowed for the model to give negative results when the previous model did not. It also allowed for a higher maximum than the non-feature engineered model did. While the Mean Square Error increased doing this, the model more accurately followed what happens in a game in real life.
Passing Result
The passing result model was done with a standard LinearRegression model. Very little feature engineering was done here and here is the result of the descriptions of the prediction vs the actual result.

Figure 6. Statistics of the train and test data frames to predict a pass play

This model does follow the quartiles and mean very well but it does not follow the explosive or negative plays as well as the rushing model does.
4th Down Classifier
This classifier is utilized to determine what the offense will do on a fourth down. Figure 7 shows the most common plays on fourth down.


Figure 7. Percentages of each type of play ran on 4th down

As shown in Figure 7, punting was the most common fourth down play. A GradientBoostRegression model trained on this set of data and was shown all of the offensive stats for the offense team. The feature importance for this model is shown below in Figure 8.


Figure 8. Feature importance when a play is ran on 4th down

The regression model primarily keys off of what yardLine the team was at as well as how far they are from a first down. To determine how accurate this model was a confusion matrix in Figure 9 was made.


Figure 9. Confusion matrix to decide what play will be ran on 4th down

The classifier is accurate on three of the four options for a fourth down. It almost always correctly guesses the two most common as well as usually selects the rushing option. This shows that the model is close to realistic decisions made in the same situation.
Field Goal Classifier
The field goal classifier is a binary classifier that determines whether or not a field goal was made or if it was missed. This data is the first three weeks of 2016, combining the ESPN CFB dataset as well as the special teams data from Sports Reference. The three features that were the most important were the yardline the ball was on, the number of field goals made for the offense, and the field goal percentage that the offense had.

Figure 10: Feature importance for field goal attempts with gradient boosting classifier
 
From Figure 10, the most important feature was the yard line of the play which closely follows standard strategy in American football.
Generate Drive

Figure 11. Code to determine drive results for a team

The code seen in Figure 11 called functions that utilize all the models so far described and then the final function, `predict_winner’, used the Predict Winner by Quarter model that is described in the next section. Figure 12 below describes in a situation for a team starting on the 99 yard line, the next two plays will be a running play and then the offense team (Texas) has a 91% chance to win.

Figure 12: Example of plays in a generated drive

Another example from the same game showed the offense team (Notre Dame this time), not scoring a touchdown.


Figure 13: Example play that shows the play type, resulting score, and chance of winning the game

This time the offense threw a pass but did not get a touchdown. Instead, they kick a field goal and only have a 6% chance of winning the game. The ultimate result of this game was that Texas wins over Notre Dame.

Predict Winner by Quarter
While the plays and drives are predicted and the football game plays, the prediction of the game winner is also predicted. The models for predicting the winner are split by quarters to account for game situations and strategies changing per quarter. The three methods tried for modeling the winner by quarter were XGBoost, Random Forest, and Gaussian Naive Bayes (NB).

Close to 20,000 rows per quarter (multiple games and hundreds of players per game) were used to compile the data can be seen in Figure 14. The differences in total number of wins and losses is due to the number of plays ran each quarter.


Figure 14. Number of data points in each quarter for the home team

Dataset features were engineered to include the following: 

'homeScore', 'awayScore',  'down', 'distance', 'Passing Completion', 'Pass Attempts',  'completion percentage', 'passing yards', 'Passing Touchdowns', 'Rush Attempts', 'Rushing yards', 'rush average', 'Rush Touchdowns', 'Total Offense Plays', 'Total yards', 'total yards per play', 'First down by pass', 'First down by rush', 'Number of Penalties',  'opponent completions', 'opponent attempts',  'opponent completion percentage', 'opponent passing yards',  'opponent passing touchdowns', 'opponent rush attempts', ‘'opponent rush yards', 'opponent average rush',  'opponent rush touchdowns', 'total opponent plays', 'total opponent yards', 'opponent total yards per play',  'opponent pass first downs', 'opponent rush first downs'

Some of these features came from the ESPN CFB play by play data, and some came from the Sport Reference stats set. The initial ESPN CFB dataset was joined together by Offensive Stats from the Sports Reference stats set on the “offenseTeam” field and the Defensive Stats from the Sports Reference stats on the “defenseTeam” field. This allowed us to incorporate the statistics of both teams in their current state of play on the field. 

As seen in Figure 15, XGBoost was most accurate for the 3rd and 4th quarters while GaussianNB was most accurate for the 1st and 2nd quarters. These models were chosen to be used with the drive generator model. Overall, the highest accuracy was about 0.62 from XGBoost and the 3rd quarter and Gaussian NB in the 2nd quarter. The expected trend was to have high accuracy of predicting the game winner as the game progressed. However in both the Random Forest and GaussianNB models, the prediction peaked at the end of the first half and trended downwards to the end of the game. 


Figure 15. Comparison of model accuracies per quarter




Figure 16. XGBoost feature importance by quarter

As seen in Figure 16, the feature importance reported by XGboost changed as the game progressed. In the first quarter, the most important features were completion percentage, opponent rush attempts and first down by rush. These are all ways that players move the ball down the field and keep a drive going. In the third and fourth quarter, the home and away score became the most important features. Typically, when a team has a higher score as the game gets closer to ending, the team will prefer rush plays to continue running the clock. If they were losing, they would prefer pass plays to risk more possible bad plays in return for larger yard gains per play and more chances to stop the clock.
Future Improvements
While the model produces some ability to predict game winners, there are some methods that might have improved the overall model:
Link not just team data, but player data as well
Player data could lead to more accurate predictions because college football teams can completely change team rosters within 2-3 seasons.
Add a momentum feature where a team has had a string of successful plays which makes their win more likely
Another feature of sports is that a team can gain momentum after a string of good plays that can go against their “team average” statistics. The model in this project mostly plays off the team average of the entire season.
Add turnovers and account for more complicated play results such as penalties
The play features were stripped to result in only rush and pass plays without accounting for turnovers which could lead to great field position or points for the other team. The simple nature of the model might have added to the inaccuracy of the winner prediction.
Conclusion
The highest model accuracy achieved was about 0.6. While this seems low, sport games by their nature are hard to predict as anything can happen for an upset, where the team that is not expected to win wins, to happen. It is hopeful that the model did produce a better result than guessing between the two teams, but there is a lot of room for improvement. Including more data sets such as specific player history and making the model more complex by adding momentum and accounting for team mistakes
