#!/usr/bin/env python
# coding: utf-8

# In[104]:


import pandas as pd


# In[105]:


batting = pd.read_csv("ODIBatting.csv")
batting.head(1)


# In[106]:


batting = batting.rename(columns={"Career End" : "Career_End", "Runs Scored" : "Runs", "Highest Innings Score Num" : "Highest Score", "Hundreds Scored" : "Centuries", "Ducks Scored" : "Ducks", "Batting Strike Rate" : "Strike Rate", "Scores Of Fifty Or More" : "Half Centuries"})
batting = batting.drop(batting[batting.Career_End < 2018].index)
batting.head()


# In[107]:


battingFeatures = batting.drop(columns=['Career Span', 'Career Start', 'Career_End', 'Highest Innings Score', 'Player Count', '40+ Batting Avg', 'Country', '90+ Batting Strike Rate', '5000+ Runs Scored' ])
def removeYear(x):
    return x[:-12]
def removeNA(x):
    if(x == '-'):
        return 0
    return x

battingFeatures["Player"] = battingFeatures["Player"].apply(removeYear)
battingFeatures["Innings Batted"] = battingFeatures["Innings Batted"].apply(removeNA)
battingFeatures["Not Outs"] = battingFeatures["Not Outs"].apply(removeNA)
battingFeatures["Runs"] = battingFeatures["Runs"].apply(removeNA)
battingFeatures["Highest Score"] = battingFeatures["Highest Score"].apply(removeNA)
battingFeatures["Batting Avg"] = battingFeatures["Batting Avg"].apply(removeNA)
battingFeatures["Balls Faced"] = battingFeatures["Balls Faced"].apply(removeNA)
battingFeatures["Strike Rate"] = battingFeatures["Strike Rate"].apply(removeNA)
battingFeatures["Centuries"] = battingFeatures["Centuries"].apply(removeNA)
battingFeatures["Half Centuries"] = battingFeatures["Half Centuries"].apply(removeNA)
battingFeatures["Ducks"] = battingFeatures["Ducks"].apply(removeNA)

battingFeatures["Player"] = battingFeatures["Player"].astype(str)
battingFeatures["Innings Batted"] = pd.to_numeric(battingFeatures["Innings Batted"])
battingFeatures["Not Outs"] = pd.to_numeric(battingFeatures["Not Outs"])
battingFeatures["Runs"] = pd.to_numeric(battingFeatures["Runs"])
battingFeatures["Highest Score"] = pd.to_numeric(battingFeatures["Highest Score"])
battingFeatures["Batting Avg"] = pd.to_numeric(battingFeatures["Batting Avg"])
battingFeatures["Balls Faced"] = pd.to_numeric(battingFeatures["Balls Faced"])
battingFeatures["Strike Rate"] = pd.to_numeric(battingFeatures["Strike Rate"])
battingFeatures["Centuries"] = pd.to_numeric(battingFeatures["Centuries"])
battingFeatures["Half Centuries"] = pd.to_numeric(battingFeatures["Half Centuries"])
battingFeatures["Ducks"] = pd.to_numeric(battingFeatures["Ducks"])


battingFeatures.head()


# In[108]:


battingFeatures.shape


# In[109]:


battingFeatures.dtypes


# In[110]:


battingFeatures = battingFeatures.sort_values(by=['Runs'], ascending=False)
battingFeatures.head()


# In[111]:


battingFeatures = battingFeatures.reset_index(drop=True)
battingFeatures.head()


# In[112]:


temp =pd.read_csv("battingFeaturesRating.csv")
temp.head()
battingFeatures["Rating"] = temp["Rating"]
battingFeatures = battingFeatures.sort_values(by=['Rating'], ascending=False)
battingFeatures = battingFeatures.reset_index(drop=True)
battingFeatures


# In[113]:


#Count number of rated players
ratedPlayerCount = len(battingFeatures[battingFeatures.Rating > 0].index)

#Define training and testing data
X_train = battingFeatures.drop(columns=['Player', 'Rating'])
X_test = X_train.iloc[ratedPlayerCount:,:]
X_train = X_train.iloc[:ratedPlayerCount,:]

#store ratings in y_train
y_train = battingFeatures['Rating']
y_train = y_train.iloc[:ratedPlayerCount]


# In[114]:


#Random Forest
from sklearn.ensemble import RandomForestRegressor 

regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
print(regr.feature_importances_)
RFresults = regr.predict(X_test)
'''
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)
RFresults = clf.predict(X_test)'''

battingFeatures["Rating"][ratedPlayerCount:] = RFresults


# In[115]:


battingFeatures = battingFeatures.sort_values(by=['Rating'], ascending=False)
battingFeatures = battingFeatures.reset_index(drop=True)
battingFeatures


# In[116]:


bowling=pd.read_csv("bowlingFeatureFinal.csv")
bowling


# In[128]:


Final = pd.merge(battingFeatures, bowling, how='inner', on = 'Player')
Final=Final.drop(columns=['Matches Played','Innings Batted','Not Outs','Runs','Highest Score','Batting Avg','Balls Faced','Strike Rate','Centuries','Half Centuries','Ducks'])
Final=Final.drop(columns=['Innings Bowled In','Balls Bowled','Runs Conceded','Wickets Taken','Bowling Avg','Economy Rate','Bowling Strike Rate'])
Final=Final.rename(index=str, columns={"Rating": "Batting Ratings", "Ratings": "Bowling Ratings"})
Final['Overall Rating']=Final['Batting Ratings']+Final['Bowling Ratings']
Final=Final.sort_values(by=['Overall Rating'], ascending=False)
Final


# In[130]:


import random
Teams={'CSK':[],'MI':[],'RCB':[],'SRH':[],'KXIP':[],'DC':[],'KKR':[],'RR':[]}
Team=[]
FinalList = list(Final['Player'])
count=0
#Snake Draft implementation
for key in Teams:
    Team.append(key)
for i in range(0,6):
    random.shuffle(Team)
    for j in Team:
        Teams[j].append(FinalList[count])
        count=count+1
    print("Order of selection by Snake Draft System is",Team)

    for j in reversed(Team):
        Teams[j].append(FinalList[count])
        count=count+1
    print("Order of selection by Snake Draft System is",Team[::-1])

   


            


# In[119]:


FinalTeams=pd.DataFrame.from_dict(Teams)
FinalTeams


# In[ ]:




