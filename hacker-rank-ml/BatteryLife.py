import sys
from sklearn.linear_model import LinearRegression
import numpy as np

training_data = {}

with open("battery_life_data.txt","r") as file:
    instance = []
    for line in file:
        if len(line.strip())>0:
            instance = line.rstrip('\n').split(",")
            training_data[instance[0]] = instance[1]
print (training_data)