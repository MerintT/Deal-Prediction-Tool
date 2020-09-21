import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

##Data Collection Stage
# In this stage we collate all data collected through various SQL queries on Redash.
#This includes Main data frame along with Chats, Pickup and Dropoff Frequencies, Requester and Addressee Frequencies,
# response times and others.

# This CSV file is the Main dataframe collected through SQL
df_Main = pd.read_csv('Main6.csv')
df_Main = df_Main.fillna(0) # Here we replacce NaNs with 0

# As the rest of the data is stored on Google sheets through SQL queries, it needs to be loaded on Python.

# Therefore, Define a function to load the data from Google sheets to Python.
#This function can then be called to load any data from Google sheets.
def dataGS(gs,ws):
	googleSheetId = gs
	worksheetName = ws
	URL = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
		googleSheetId,
		worksheetName
	)
	df_GS = pd.read_csv(URL)
	df_GS = pd.DataFrame(df_GS)
	return df_GS

# Loading Chats dataset through the dataGS funtion created above
df_Chats = dataGS(gs ='Google Sheet Link', ws= 'Chats')

# Loading Dropoff frequency dataset through the dataGS funtion created above
df_drop = dataGS(gs ='Google Sheet Link', ws= 'Dropoff')
df_drop = df_drop.rename(columns={'Dropoff': 'Dropoff_location'}) #Renaming columns as per Main dataset

#  Loading Pickup frequency dataset through the dataGS funtion created above
df_pick = dataGS(gs ='Google Sheet Link', ws= 'Pickup')
df_pick = df_pick.rename(columns={'Pickup': 'Pickup_location'}) # Renaming column as per main dataset

#  Loading Requester frequency dataset through the dataGS funtion created above
df_requester = dataGS(gs ='Google Sheet Link', ws= 'Requester')
df_requester = df_requester.rename(columns={'requester_id': 'Requester'}) # Renaming column as per main dataset

#  Loading Addressee frequency dataset through the dataGS funtion created above
df_add = dataGS(gs ='Google Sheet Link', ws= 'Addressee')
df_add = df_add.rename(columns={'Addressee_id': 'Addressee'})# Renaming column as per main dataset

#  Loading Chat response time dataset through the dataGS funtion created above
Resp_chat = dataGS(gs ='Google Sheet Link', ws= 'Chat_Response')

#  Loading Edit response time dataset through the dataGS funtion created above
Resp_version = dataGS(gs ='Google Sheet Link', ws= 'Version_Response')

# Now that the different datasets have been loaded, they need to be merged into a single training dataset
# Define a function to merge datasets and can be called each time datasets need to be merged.
def dataMerge(df1,df2,col):
	result = pd.merge(df1, df2, how='left', on=[col])
	return result

#Merging Main and Chats
result1 = dataMerge(df1=df_Main,df2=df_Chats,col='Request_id')
result1 = result1= result1.fillna(0)

#Merging Pickup frequency
result1 = dataMerge(df1=result1,df2=df_pick,col='Pickup_location')
result1.drop(["Pickup_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result1 = result1.rename(columns={'Shares_accepted': 'Pickup_location'}) # Renaming columns

#Merfing Dropoff frequency
result2 = dataMerge(df1=result1,df2=df_drop,col='Dropoff_location')
result2.drop(["Dropoff_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result2 = result2.rename(columns={'Shares_accepted': 'Dropoff_location'}) # Renaming columns

#Merfing Requester frequency
result3 = dataMerge(df1=result2,df2=df_requester,col='Requester')
result3.drop(["Requester","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result3 = result3.rename(columns={'Frequency': 'Requester'})# Renaming columns

#Merfing Addressee frequency
result4 = dataMerge(df1=result3,df2=df_add,col='Addressee')
result4.drop(["Addressee","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result4 = result4.rename(columns={'Frequency': 'Addressee'})# Renaming columns
result4 = result4.fillna(0)

#Merfing Chat Response time
result5 = dataMerge(df1=result4,df2=Resp_chat,col='Request_id')

# Merfing Edit Response time
result6 = dataMerge(df1=result5,df2=Resp_version,col='Request_id')

result7= result6

# Replacing NaNs for Chat response time with mean value
result7['Chat_Response'].fillna((result7['Chat_Response'].mean()), inplace=True)

# Converting Pickup  charges feature into Binary feature.
# If there is a Pickup charge it is marked as 1 else 0.
result7.loc[result7['Pickup_Charges'] > 0, 'Pickup_Charge'] = 1
result7.loc[result7['Pickup_Charges'] == 0, 'Pickup_Charge'] = 0

# Removing all unwanted columns to create the final training dataset
result7.drop(['Request_id','Created_date', 'Container_Type', 'Pickup_Charges', 'Chat_Response', 'Edit_ResponseTime'], axis=1 , inplace=True)

# As Edits is the version when result is obtained and for all deals version 1 is when the deal is created.
# Therefore version 1 for all deals is not an edit.
# 1 is subtracted from Edits column for all requests
result7['Edits'] = result7['Edits'] - 1

# One hot encoding the Direction feature. 
# As Direction feature has just 2 classes (Using and Supplying), it is encoded accordingly
result7 = pd.concat([result7,pd.get_dummies(result7['Direction'])],axis=1)
result7.drop(['Direction'],axis=1, inplace=True) # Remove the old column

# One hot encoding the Sources feature.
# This feature has total of 16 classes and is encoded accordingly
result7 = pd.concat([result7,pd.get_dummies(result7['Sources'])],axis=1)
result7.drop(['Sources'],axis=1, inplace=True) # Remove the old column

# Reorder the columns as required such that the dependent variable is kept last.
order = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,7] # setting column's order
result7 = result7[[result7.columns[i] for i in order]]

# Dropping another unwanted column from the Sources.
result7.drop(['RAPID_BOOKING'], axis=1 , inplace=True)
#print(result7)

## Balancing the Dataset

# Count the dependent variable class counts
result7.Deal_status.value_counts()

# Split both the classes into separate dataframes
result7_majority = result7[result7.Deal_status==0]
result7_minority = result7[result7.Deal_status==1]

# Check number of datapoints with lower class variable (Successful in this case) 
samples = result7_minority.shape[0]

# Downsample the upper class to meet the lower class count
result7_majority_downsampled = resample(result7_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=samples,     # to match minority class
                                 random_state=123)

result7_downsampled = pd.concat([result7_majority_downsampled, result7_minority])
print(result7_downsampled.Deal_status.value_counts())

##Data Analysis

#Setting the grid for xgboost classifier
grid = {"min_child_weight":[1],
            "max_depth":[3,7,9,10],
            "learning_rate":[0.08,0.09,0.1,0.3,0.4],
            "reg_alpha":[0.01,0.02,0.3],
            "reg_lambda":[0.8,0.9,1,1.2,1.4],
            "gamma":[0.002,0.004,0.03],
            "subsample":[0.9,1.0],
            "colsample_bytree":[0.7,0.9,1.0],
            "objective":['binary:logistic'],
            "nthread":[-1],
            "scale_pos_weight":[1],
            "seed":[0,10,42,13],
            "n_estimators": [50,100,200,300,400,500]}

# If MLP was the classifier then the grid would be the following
# gridMLp = {'activation' : ['logistic'],
#                'max_iter':[7000],
#                'random_state':[0,10,42,13],
#                'solver' : ['adam'],
#                'alpha': [0.0001],
#                'hidden_layer_sizes': [(100),(100,100),
#                                        (100,100,100),
#                                     (100,100,100,100),
#                                    (100,100,100,100,100),
#                                    (145),(150),(160),
#                                    (170),(50)]
#           }

#Defininig timer for hiperparameter selection to see the computational efficiency of Hyperparameter tuning
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#Calling the xgboost classifier
classifier=xgboost.XGBClassifier()

#Random seach for Hiperparameters
random_search=RandomizedSearchCV(classifier,
	param_distributions=grid,
	n_iter=5,
	scoring='roc_auc',
	n_jobs=-1,
	cv=5,  #Setting cross-validation folds to 5. It can be changed, however, it would be computationally heavy
	verbose=3)

# We can also use grid search for Hyperparameter tuning. However, it's exhaustively computationally heavy as
# compared to Randomized Search. 
# In this case, we first tried with the Grid Search. However, our Randomized Search results were not very different
# from the Grid Search and therefore, we stuck to Randomized Search for better coding efficiency.

# If we're using Grid Search, the following code needs to be followed:

#grid_search =GridSearchCV(estimator=classifier,
#  param_grid=grid,
#  scoring='accuracy',
# cv=5,
# n_jobs=-1)

# Splitting the Dependent variable from the rest of the dataset
X=result7_downsampled.iloc[:,0:30]
Y=result7_downsampled.iloc[:,30]
#print(Y)

# Running the timer module defined earlier
from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
print(timer(start_time)) # timing ends here for "start_time" variable

# Printing the best parameters estimated by the Random/Grid Search CV
best= random_search.best_estimator_
print(best)

# Using the best Hyperparameters tuned earlier and training the classifier model
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.002,
              learning_rate=0.3, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=None, n_estimators=300, n_jobs=1,
              nthread=-1, objective='binary:logistic', random_state=0,
              reg_alpha=0.3, reg_lambda=0.8, scale_pos_weight=1, seed=0,
              silent=None, subsample=0.9, verbosity=1)

# Setting empty array for CV accuracy scores
accuracy = []

# Using Stratified K fold CV to get equal classes 
skf = StratifiedKFold(n_splits=10, random_state = None)
skf.get_n_splits(X,Y)

# Fitting the trained classifier model to the training dataset
for train_index, test_index in skf.split(X,Y):
  print("Train",train_index,"Validation", test_index)
  X_train,X_test = X.iloc[train_index], X.iloc[test_index]
  Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]

  classifier.fit(X_train,Y_train)
  prediction = classifier.predict(X_test)
  score = accuracy_score(prediction,Y_test)
  accuracy.append(score)

# Printing the Accuracy of the model
print(accuracy)
print(np.array(accuracy).mean())

#clf = classifier.fit (X,Y)

# Storing the trained classifier as a Pickle file for future use.
with open('model_pkl', 'wb') as f:
	pickle.dump(classifier,f)