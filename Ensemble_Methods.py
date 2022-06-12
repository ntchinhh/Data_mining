# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:20:04 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 02:09:15 2021

@author: Admin
"""
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

import random
import numpy as np
import math
import sklearn
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
# instantiate labelencoder object
from sklearn.tree import export_text

from IPython.display import Image  
from sklearn.tree import export_graphviz



    
def stratify_sampling(df):
    r,c = df.shape
    X = df.iloc[:,df.columns !='class']
    features = X.columns.tolist()
    y = df[['class']]

    acc_RF= list()
    acc_Bagging = list()
    acc_Boosting = list()
   
    time_RF =list()
    time_Bagging =list()
    time_Boosting =list()
    
    for i in range (100):
        Train_x, Test_x, Train_y, Test_y = train_test_split(X,  y, stratify=y, test_size=0.4, random_state = random.randint(0,100000))
        
#random forest
        start = datetime.now()

        rf_model = RandomForestClassifier(n_estimators=100, max_features= int(math.sqrt(c))+1)

        rf_model.fit(Train_x,Train_y.values.ravel())
        pred_y = rf_model.predict(Test_x)
        end = datetime.now() -start
        time_RF.append(end)
        
        print("Accuracy RandomForest:",metrics.accuracy_score(Test_y, pred_y))
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))

       # print(metrics.confusion_matrix(Test_y,predictions))
       # print(metrics.classification_report(Test_y,predictions))
        #print(metrics.accuracy_score(Test_y, predictions))
    
#Bagging    
        start = datetime.now()        
        base_cls = DecisionTreeClassifier()                
        model_bagging = BaggingClassifier(base_estimator = base_cls, n_estimators = 100)        
        model_bagging.fit(Train_x, Train_y.values.ravel())
        predictions = model_bagging.predict(Test_x)
        
        end = datetime.now() -start
        time_Bagging.append(end)
                
        acc = metrics.accuracy_score(Test_y, predictions)    
        acc_Bagging.append(acc)        
        print ("Bagging: ",acc)     
        
#Boosting    
        start = datetime.now()            
        model_boosting = GradientBoostingClassifier(n_estimators = 100)        
        model_boosting.fit(Train_x, Train_y.values.ravel())
        predictions = model_boosting.predict(Test_x)
        
        end = datetime.now() -start
        time_Boosting.append(end)
                
        acc = metrics.accuracy_score(Test_y, predictions)    
        acc_Boosting.append(acc)        
        print ("Boosting: ",acc)     

    

    print ("Độ Chính Xác Random forest: ", sum(acc_RF))
    print ("Độ Chính Xác Trung Bình Bagging: ", sum(acc_Bagging))
    print ("Độ Chính Xác Trung Bình Boosting: ", sum(acc_Boosting))
    
    print ("Thời gian Random forest: ", np.mean(time_RF))
    print ("Thời gian Bagging : ", np.mean(time_Bagging))
    print ("Thời gian Boosting : ", np.mean(time_Boosting))
    
    
    results =[]
    results.append(acc_RF)
    results.append(acc_Bagging)
    results.append(acc_Boosting)



    names =('Random forest','Bagging', 'Boosting')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy') 
    
    return 

    
def readData(df):
    r, c = df.shape
    size_sample = int(0.7*r)
    print (size_sample)    
    
    row_id  = random.sample(range(r), r)   
    acc_RF= list()
    acc_Bagging =list()
    acc_Boosting = list()
    for i in range(100):
        print (i)
     
        chosen_idx = np.random.choice(r, replace = False,  size = size_sample) 
        row_y = list(set(row_id) - set(chosen_idx))
    
        # train va test
        Train = df.iloc[chosen_idx]
        Test  = df.iloc[row_y]
        Train_x = Train.iloc[:,Train.columns !='class']
        Train_y = Train.loc[:,'class']
        #print (Train_x)
        
        Test_x = Test.iloc[:,Test.columns !='class']
        Test_y = Test.loc[:,'class']
       
        rf_model = RandomForestClassifier(n_estimators=100,max_features= int(math.sqrt(c))+1)
    
        rf_model.fit(Train_x,Train_y)
        pred_y = rf_model.predict(Test_x)
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))
        print("Accuracy RF:",metrics.accuracy_score(Test_y, pred_y))
        
        print(metrics.confusion_matrix(Test_y,pred_y))
        print(metrics.classification_report(Test_y,pred_y))
        print(metrics.accuracy_score(Test_y, pred_y))

        bagging_model = BaggingClassifier()()
        
        bagging_model.fit(Train_x, Train_y)
    
        predictions = bagging_model.predict(Test_x)
        acc = metrics.accuracy_score(Test_y, predictions)
        acc_Bagging.append(acc)
        print("Accuracy Bagging:",metrics.accuracy_score(Test_y, predictions))
        print(metrics.confusion_matrix(Test_y,predictions))
        print(metrics.classification_report(Test_y,predictions))
        #print(metrics.accuracy_score(Test_y, predictions))
        
        boosting_model = GradientBoostingClassifier()()
        
        boosting_model.fit(Train_x, Train_y)
    
        predictions = boosting_model.predict(Test_x)
        acc = metrics.accuracy_score(Test_y, predictions)
        acc_Boosting.append(acc)
        print("Accuracy Boosting:",metrics.accuracy_score(Test_y, predictions))
        print(metrics.confusion_matrix(Test_y,predictions))
        print(metrics.classification_report(Test_y,predictions))
        #print(metrics.accuracy_score(Test_y, predictions))
    
    results =[]
    results.append(acc_RF)
    results.append(acc_Bagging)
    results.append(acc_Boosting)
  
    names =( 'Random forest', 'Bagging', 'Boosting')
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')

    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy') 
    
    print ("Độ Chính Xác Random forest: ", sum(acc_RF))
    print ("Độ Chính Xác Trung Bình Bagging: ", sum(acc_Bagging))
    print ("Độ Chính Xác Trung Bình Boosting: ", sum(acc_Boosting))
    return 

    
def main():
    df = pd.read_csv('C:\\Users\\Admin\\Desktop\\Data\\covtype.csv' ) 
    df.columns.values[0] = "class"   
   # readData(df)    
    stratify_sampling(df)
    return

if __name__ == "__main__":
    main()
        
