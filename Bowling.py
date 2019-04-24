#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


bowling = pd.read_csv("ODIBowling.csv")
bowling.head()


# In[3]:


bowlingFeatures = bowling.drop(columns=['Best Bowling In An Innings', 'Four Wickets In An Innings', 'Five Wickets In An Innings', '200+ Wickets Taken', '<35.00 Bowling Avg', '<4.00 Economy Rate', '<40.00 Bowling Strike Rate' ])
bowlingFeatures.head()
def removeYear(x):
    return x[:-12]
def removeNA(x):
    if(x == '-'):
        return 0
    return x
bowlingFeatures["Player"] = bowlingFeatures["Player"].apply(removeYear)
bowlingFeatures["Innings Bowled In"] = bowlingFeatures["Innings Bowled In"].apply(removeNA)
bowlingFeatures["Balls Bowled"] = bowlingFeatures["Balls Bowled"].apply(removeNA).fillna(0)
bowlingFeatures["Runs Conceded"] = bowlingFeatures["Runs Conceded"].apply(removeNA)
bowlingFeatures["Wickets Taken"] = bowlingFeatures["Wickets Taken"].apply(removeNA)
bowlingFeatures["Bowling Avg"] = bowlingFeatures["Bowling Avg"].apply(removeNA)
bowlingFeatures["Economy Rate"] = bowlingFeatures["Economy Rate"].apply(removeNA)
bowlingFeatures["Bowling Strike Rate"] = bowlingFeatures["Bowling Strike Rate"].apply(removeNA)
bowlingFeatures["Player"] = bowlingFeatures["Player"].astype(str)
bowlingFeatures["Innings Bowled In"] = pd.to_numeric(bowlingFeatures["Innings Bowled In"])
bowlingFeatures["Balls Bowled"] = pd.to_numeric(bowlingFeatures["Balls Bowled"])
bowlingFeatures["Runs Conceded"] = pd.to_numeric(bowlingFeatures["Runs Conceded"])
bowlingFeatures["Wickets Taken"] = pd.to_numeric(bowlingFeatures["Wickets Taken"])
bowlingFeatures["Bowling Avg"] = pd.to_numeric(bowlingFeatures["Bowling Avg"])
bowlingFeatures["Economy Rate"] = pd.to_numeric(bowlingFeatures["Economy Rate"])
bowlingFeatures["Bowling Strike Rate"] = pd.to_numeric(bowlingFeatures["Bowling Strike Rate"])

bowlingFeatures.head()


# In[4]:


batting=pd.read_csv("battingFeaturesRating.csv")
batting.head()
batting.shape


# In[5]:


bowlingFeatureFinal = pd.merge(batting, bowlingFeatures, how='inner', on = 'Player')
bowlingFeatureFinal=bowlingFeatureFinal.drop(columns=['Matches Played','Innings Batted','Not Outs','Runs','Highest Score','Batting Avg','Balls Faced','Strike Rate','Centuries','Half Centuries','Ducks','Rating'])
bowlingFeatureFinal = bowlingFeatureFinal.sort_values(by=['Wickets Taken'], ascending=False)
bowlingFeatureFinal = bowlingFeatureFinal.reset_index(drop=True)
bowlingFeatureFinal.head()
temp =pd.read_csv("bowlingFeatures.csv")
temp['Ratings']=temp['Ratings'].fillna(0)
bowlingFeatureFinal["Ratings"] = temp["Ratings"]
bowlingFeatureFinal["Ratings"]=bowlingFeatureFinal["Ratings"].astype(int)
bowlingFeatureFinal = bowlingFeatureFinal.sort_values(by=['Ratings'], ascending=False)
bowlingFeatureFinal = bowlingFeatureFinal.reset_index(drop=True)
bowlingFeatureFinal


# In[6]:


ratedPlayerCount = len(bowlingFeatureFinal[bowlingFeatureFinal.Ratings > 0].index)


#Define training and testing data
X_train = bowlingFeatureFinal.drop(columns=['Player', 'Ratings'])
X_test = X_train.iloc[ratedPlayerCount:,:]
X_train = X_train.iloc[:ratedPlayerCount,:]

#store ratings in y_train
y_train = bowlingFeatureFinal['Ratings']
y_train = y_train.iloc[:ratedPlayerCount]


# In[7]:


from sklearn.ensemble import RandomForestRegressor 

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
print(regr.feature_importances_)
RFresults = regr.predict(X_test)
'''
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)
RFresults = clf.predict(X_test)'''

bowlingFeatureFinal["Ratings"][ratedPlayerCount:] = RFresults


# In[8]:


bowlingFeatureFinal = bowlingFeatureFinal.sort_values(by=['Ratings'], ascending=False)
bowlingFeatureFinal = bowlingFeatureFinal.reset_index(drop=True)
bowlingFeatureFinal.to_csv('bowlingFeatureFinal.csv',index=False)

