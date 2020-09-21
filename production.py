import numpy as np
import pandas as pd
import pickle

#Loading the Pickle file of the classifier trained in Deal_forecaster.py
with open('model_pkl', 'rb') as f:
	mp = pickle.load(f)

#Function to load data from Google sheets
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

#Loading Data as similar to Training dataset
Deals = dataGS(gs ='Google Sheet Link', ws= 'Deal3')


df_Chats = dataGS(gs ='Google Sheet Link', ws= 'Chats')

df_drop = dataGS(gs ='Google Sheet Link', ws= 'Dropoff')
df_drop = df_drop.rename(columns={'Dropoff': 'Dropoff_location'})

df_pick = dataGS(gs ='Google Sheet Link', ws= 'Pickup')
df_pick = df_pick.rename(columns={'Pickup': 'Pickup_location'})

df_requester = dataGS(gs ='Google Sheet Link', ws= 'Requester')
df_requester = df_requester.rename(columns={'requester_id': 'Requester'})

df_add = dataGS(gs ='Google Sheet Link', ws= 'Addressee')
df_add = df_add.rename(columns={'Addressee_id': 'Addressee'})

Resp_chat = dataGS(gs ='Google Sheet Link', ws= 'Chat_Response')

Resp_version = dataGS(gs ='Google Sheet Link', ws= 'Version_Response')

#Function for Merging Data set
def dataMerge(df1,df2,col):
	result = pd.merge(df1, df2, how='left', on=[col])
	return result

#Priliminary Processing of data according to the trained dataset
Deal1 = dataMerge(df1=Deals,df2=df_Chats,col='Request_id')
Deal1 = Deal1= Deal1.fillna(0)

Deal1 = dataMerge(df1=Deal1,df2=df_pick,col='Pickup_location')
Deal1.drop(["Pickup_location","Count", "when_accepted"],inplace=True,axis=1)
Deal1 = Deal1.rename(columns={'Shares_accepted': 'Pickup_location'})

Deal2 = dataMerge(df1=Deal1,df2=df_drop,col='Dropoff_location')
Deal2.drop(["Dropoff_location","Count", "when_accepted"],inplace=True,axis=1)
Deal2 = Deal2.rename(columns={'Shares_accepted': 'Dropoff_location'})

Deal3 = dataMerge(df1=Deal2,df2=df_requester,col='Requester')
Deal3.drop(["Requester","Count", "when_accepted"],inplace=True,axis=1)
Deal3 = Deal3.rename(columns={'Frequency': 'Requester'})

Deal4 = dataMerge(df1=Deal3,df2=df_add,col='Addressee')
Deal4.drop(["Addressee","Count", "when_accepted"],inplace=True,axis=1)
Deal4 = Deal4.rename(columns={'Frequency': 'Addressee'})
Deal4 = Deal4.fillna(0)

Deal5 = dataMerge(df1=Deal4,df2=Resp_chat,col='Request_id')

Deal6 = dataMerge(df1=Deal5,df2=Resp_version,col='Request_id')

Deal6['Chat_Response'].fillna((Deal6['Chat_Response'].mean()), inplace=True)
Deal6['Edit_ResponseTime'].fillna((Deal6['Edit_ResponseTime'].mean()), inplace=True)

Deal7=Deal6
#Deal7['Location'] = Deal7[['Pickup_location', 'Dropoff_location']].mean(axis=1)
#Deal7['Partner'] = Deal7[['Requester', 'Addressee']].mean(axis=1)
Deal7.loc[Deal7['Pickup_Charges'] > 0, 'Pickup_Charge'] = 1
Deal7.loc[Deal7['Pickup_Charges'] == 0, 'Pickup_Charge'] = 0
Deal7.drop(['Request_id','Created_date', 'Pickup_Charges','Chat_Response', 'Edit_ResponseTime'], axis=1 , inplace=True)
Deal7 = Deal7.fillna(0)

Deal7 = pd.concat([Deal7,pd.get_dummies(Deal7['Direction'])],axis=1)
Deal7.drop(['Direction'],axis=1, inplace=True)

order3 = [0,1,2,3,4,5,6,22,23,24,25,26,27,28,29,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] # setting column's order
Deal7 = Deal7[[Deal7.columns[i] for i in order3]]

#Predicting data using Pickled classifier
pred_try = mp.predict(Deal7)

pred_try_df=pd.DataFrame(pred_try, columns=['Predicted_deal']) 

#Adjusting the dataframe as needed
Request = Deals.Request_id
Request = Request.reset_index(drop=True)
Request = pd.DataFrame(Request,columns=['Request_id'])

Deal_final_pred = pd.concat([Request, pred_try_df], axis=1)
Deal_final_pred.to_pickle('Prediction_pkl') #Pickling the dataframe before further modifications

CompanyOT = dataGS(gs ='Google Sheet Link', ws= 'Company')

Source = dataGS(gs ='Google Sheet Link', ws= 'Source')
Source = Source.rename(columns={'id': 'Request_id'})

Deal_final = dataMerge(df1=Deal_final_pred,df2=Source,col='Request_id')

#Exporting the results to Google sheets
import pygsheets

# Download the json file for Google API and Google Sheet API and save it in the same folder as the files.
# Enable both the APIS and then save the JSON file

# Provide authoration to the Python script using this JSON file
gc=pygsheets.authorize(service_file='API_JSON_file.json') # This is a dummy file

sh=gc.open_by_key('Any Google sheet') # Make sure the sheet has edit access
wks=sh.worksheet(property='index',value=0) # Value 0 is for first tab and 1 is for second tab

wks.set_dataframe(Deal_final, start = 'A1', copy_index=False, copy_head=True, fit=True, escape_formulae=False,nan = 0)

#Importing the results from the Pickling History file
# By doing this, it will call this file and run the entire script
from Deal_forecast_pickle_read_write import result 

wks1=sh.worksheet(property='index',value=2) # Load on another tab in the same sheet

wks1.set_dataframe(result, start = 'A1', copy_index=False, copy_head=True, fit=True, escape_formulae=False,nan = 0)
#print(Deal_final)