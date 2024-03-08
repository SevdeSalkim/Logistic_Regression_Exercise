import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read csv

dataset = pd.read_csv('odev_tenis.csv')
#print(dataset)

outlook = dataset.iloc[:,0].values.reshape(-1,1)
windy = dataset.iloc[:,3].values.reshape(-1,1)
play = dataset.iloc[:,-1].values.reshape(-1,1)
tempture_and_humidity = dataset.iloc[:,1:3].values
# OneHot Encoder
from sklearn.preprocessing import OneHotEncoder

#Object
ohe = OneHotEncoder()

outlook_encoded = ohe.fit_transform(outlook).toarray()
windy_encoded = ohe.fit_transform(windy).toarray()
play_encoded = ohe.fit_transform(play).toarray()


#Convert dataframe

outlook_df = pd.DataFrame(data=outlook_encoded,columns=["overcast","rainy","sunny"])

windy_df = pd.DataFrame(data=windy_encoded,columns=["false","isWindy"]) #Dummy variable (kukla değişken)
windy_df = windy_df.drop(["false"],axis=1)


play_df = pd.DataFrame(data=play_encoded,columns=["false","isPlay"]) #Dummy variable (kukla değişken)
play_df = play_df.drop(["false"],axis=1)

tempture_and_humidity_df = pd.DataFrame(data=tempture_and_humidity, columns=["temperature","humidity"])

# concat  merge dataframe

df = pd.concat([outlook_df,tempture_and_humidity_df,windy_df,play_df], axis=1)


# Split train and test data

from sklearn.model_selection import train_test_split
#features
X = df.iloc[:,:-1].values
#targets
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)


#Model 
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(X_train, y_train)
# tahmin(predict)
y_pred = reg.predict(X_test)

# metrics
from sklearn.metrics import accuracy_score

print(f"Score : {accuracy_score(y_test, y_pred)}")












