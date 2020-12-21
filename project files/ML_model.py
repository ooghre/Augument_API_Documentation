# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:33:56 2020

@author: User
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

    
def preprocess(data):
    #lowercasing
    data['Body'] = data["Body"].str.lower()
    
    #stop word
    #you can experiment with commenting out stopword removal from text preprocessing as this increases accuracy
    stop = stopwords.words('english')
    data['Body'] = data['Body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    #stemming
    stemmer = nltk.stem.PorterStemmer()
    data['Body'] = data['Body'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()])) # Stem every word.
    
    return data

def convert_to_BOW(data):
    document = []
    data['Body'].apply(lambda x: [document.append(x)])
    
    vectorizer = CountVectorizer()
    data_BOW = vectorizer.fit_transform(document)
    df = pd.DataFrame(data_BOW.toarray())
    data = pd.concat([data, df], axis=1)
    return data

def classifier(data):
    data = add_length(data)
    data = preprocess(data)
    data = convert_to_BOW(data)
    model = 'with new features and with preprocessing'
    use_RandomForestClassifier(data, model)
    
def use_RandomForestClassifier(data, model):
    data = data.drop(columns = ['Body'])
    # using 10 trees
    
    column_names = list(data.columns)
    column_names.remove('Label')
    Y = data['Label']
    X = data[column_names]
    
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.6, random_state=1)
    modelRandomForest = RandomForestClassifier(random_state =1, n_estimators = 100)
    print(np.any(np.isnan(x_train))) #and gets False
    print(np.all(np.isfinite(x_train))) #and gets True
    print(np.any(np.isnan(y_train))) #and gets False
    print(np.all(np.isfinite(y_train))) #and gets True
    modelRandomForest.fit(x_train, y_train)
        
    yPredRandomForest = modelRandomForest.predict(x_test)
    print(f'for model {model} the classification report is as follows')

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(y_test, yPredRandomForest, digits=3))

def add_length(data):
    '''
    This method adds the length of the body post as a feature
    '''
    length = []
    for index, row in data.iterrows():
        length.append( len(data['Body'][index]))
    data['Length'] = length
    return data
    
def main():
    path = "data_set_annotated.xlsx"
    #read data
    data = pd.read_excel(path)
    
    data = data.drop(columns = [ 'Unnamed: 0', 'Id', 'PostTypeId', 'ParentId', 'CreationDate',
       'DeletionDate','LastEditDate', 'LastActivityDate',
       'Title'])
    classifier(data)
    
    print(data.columns)
    
if __name__ == "__main__":
    main()