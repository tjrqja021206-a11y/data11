import cv2
import pandas as pd
from pandas.core.groupby import groupby

#ssengerId,HomePlanet,CryoSleep,Cabin,Destination,Age,VIP,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck,Name,Transported

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

for df in [train, test]:
    df["Age"]=df["RoomService"].fillna(df.groupby("Age")["RoomService"].transform("median"))
    df["Transported"]=(df["Transported"]=="True").astype(int)
print(df["Transported"].isnull().sum())