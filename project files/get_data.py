# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 01:22:13 2020

@author: User
"""
import pandas as pd
import glob, os    


df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "*.csv"))))
df = df.sample(frac=1, random_state =1)
print(df.shape)
df = df.head(30)
df["Label"] = ""
Y = df["Label"]
df = df.drop(columns = [ 'AcceptedAnswerId',    'ViewCount', 'OwnerUserId',
       'OwnerDisplayName', 'LastEditorUserId', 'LastEditorDisplayName',
        'Tags', 'AnswerCount',
        'FavoriteCount', 'ClosedDate', 'CommunityOwnedDate',
       'ContentLicense'])

df.to_excel("data_set.xlsx")

print(df.shape)

#remove html tags
#delet code tags
#use ml before cosine similarity
#for index, row in df.iterrows():
#    print(row["Body"])
#    print("next line")
    
