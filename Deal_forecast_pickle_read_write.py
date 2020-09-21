import pandas as pd
import numpy as np
df = pd.read_pickle('Prediction_pkl')
#Need the below line just for first run
#df.to_pickle('Prediction_pkl_hist') 
df1 = pd.read_pickle('Prediction_pkl_hist')
df2= pd.concat([df1, df],ignore_index=True)
df2.to_pickle('Prediction_pkl_hist')
listt= df2['Request_id'].unique()
listt = listt.tolist()
first_index = []
for i in listt:
	maxx= df2[df2.Request_id==i].first_valid_index()
	first_index.append(maxx)
	
First_value= df2.take(first_index)
First_value = First_value.rename(columns={'Predicted_deal': 'First_prediction'})
last_index = []
for i in listt:
	minn= df2[df2.Request_id==i].last_valid_index()
	last_index.append(minn)
	
Last_value = df2.take(last_index)
Last_value = Last_value.rename(columns={'Predicted_deal': 'Last_prediction'})
result = pd.merge(First_value, Last_value, how='left', on=['Request_id'])

