#Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import the dataset
dataset=pd.read_csv(r"D:\Full Stack Data Science & AI\Notes\September\4 September & 5 September\Data.csv")
dataset
#iloc==index location
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
from sklearn.impute import SimpleImputer
#SimpleImputer==module
#imputer==transformer to fill missing value in the dataset
#mean == parameter tunning
#In the simpleimputer there is no concept of mode it use most_frequent
#hyperparamete tunning include median and mode (most_frequent) strategy 
imputer=SimpleImputer(strategy="most_frequent")#hyperparameter tunning
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)



