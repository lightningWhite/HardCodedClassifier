# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:05:23 2018

@author: Daniel Hornberger
@brief: This file provides a shell for a hard-coded classifier.
It classifies the iris dataset and provides functionality to be expanded for
other uses.
"""
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import csv

# A model for obtaining predicitons. It will always prdict 'setosa'
class HardCodedModel:        
    def predict(self, data_test):
        #return an array of hard coded setosas or something
        predictions = []
        for value in data_test:
            predictions.append(0)
        return predictions

# A classifier that classifies everything as 'setosa' 
class HardCodedClassifier:    
    def fit(self, data_train, targets_train):
        model = HardCodedModel()
        return model
    
def main(): 
    # Load the iris dataset
    iris = datasets.load_iris()
    
    csv_data= []
    csv_targets = []
    
    # Provide an enum to convert names to ints
    iris_enum = {
            "setosa" : 0,
            "versicolor" : 1, 
            "virginica" : 2
            }
    
    # Obtain the iris data set from a .csv file
    with open('irisDataSet.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')    
        line_count = 0
        
        # Create a data list and a targets list from the .csv data
        for row in csv_reader:
            if line_count != 0:
                csv_data.append(row[:4])
                csv_targets.append(iris_enum[str(row[-1])])
            line_count += 1

    print("Data from CSV")
    print(csv_data) 
            
    print("\n")
    print("Targets from CSV")
    print(csv_targets)
    print("\n")
    
    # Randomize the dataset and divide the data for testing and training 
    # The following line uses the iris dataset from sklearn
    data_train, data_test, targets_train, targets_test = train_test_split(
            iris.data, iris.target, test_size = 0.30, random_state=42)
    
    # The following line uses the iris dataset as read from a .csv file
#    data_train, data_test, targets_train, targets_test = train_test_split(
#            csv_data, csv_targets, test_size = 0.30, random_state=42)
    
    print("data test, train size")
    print(len(data_train))
    print(len(data_test))
    
    print("target train, test size")
    print(len(targets_train))
    print(len(targets_test))
    
    # Obtain a classifier
#    classifier = GaussianNB() # Uncomment for actual training
    classifier = HardCodedClassifier()
    
    # Fit the data
    model = classifier.fit(data_train, targets_train)
    
    # Get the predicted targets
    targets_predicted = model.predict(data_test)
    
    # Calculate and display the accuracy
    num_predictions = len(targets_predicted)    
    correct_count = 0
    for i in range(num_predictions):
        print("predicted: {}, actual: {}".format(targets_predicted[i], targets_test[i]))
        if targets_predicted[i] == targets_test[i]:
            correct_count+=1
            
    accuracy = float(correct_count) / float(num_predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100.0))
    

main()