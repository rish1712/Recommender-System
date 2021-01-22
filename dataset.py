import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import math
start=time.time()

cols = ['userid','movieid','rating','timestamp']
df=pd.read_csv('u.data',sep='\t',names=cols)


train,test=train_test_split(df,test_size=0.01,random_state=42)
user_count=(pd.unique(df['userid'])).max()
movie_count=(pd.unique(df['movieid'])).max()
#print((test))

full_data_set=np.zeros((user_count,movie_count))

for row in df.itertuples():
    full_data_set[row[1]-1][row[2]-1]=row[3]

train_mat=np.zeros((user_count,movie_count))

for row in train.itertuples():
    train_mat[row[1]-1][row[2]-1]=row[3]


#print(train_mat)


test_data_list=[]
for row in test.itertuples():
    test_data_list.append([row[1]-1,row[2]-1,row[3]])


#print(test_data_list)

end=time.time()
def rms(rmse):

    return rmse+0.7











