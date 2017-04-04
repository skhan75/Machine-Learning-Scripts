import sys
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load Data and return features and target variables
def load_data():
    features = []
    targets = []
    with open("battery_life_data.txt", "r") as file:
        instance = []
        for line in file:
            if len(line.strip()) > 0:
                instance = line.rstrip('\n').split(",")
                features.append(float(instance[0]))
                targets.append(float(instance[1]))
    return features, targets

# Uncomment the below function to see the trend in the data
#def plot_data(x, y):
    #fig, ax = plt.subplots()
    #fit = np.polyfit(x, y, 1)
    #ax.plot(x, fit[0] * x + fit[1], color='red')
    #ax.scatter(x, y)

    #fig.show()
    #plt.show()


def extract_and_clean_data():
    # Extracting only the essential features which affects our prediction based on the plot we made
    # i.e. When battery charged is >= 4, then battery life is always 8. So these features we can neglect as this won't affect our prediction

    features, targets = load_data()
    features = np.array(features)
    targets = np.array(targets)
    features = features[features < 4]
    targets = targets[targets < 8]
    features = features[features > 0]
    targets = targets[targets > 0]

    # Creating a 2D List for loading X and y into the classifier
    X = [[f] for f in features]
    y = [[t] for t in targets]

    return np.array(X), np.array(y)


# Training the classifier using Linear Regression
def train_classifier():
    X,y = extract_and_clean_data()
    clf = LinearRegression()
    trained_data = clf.fit(X,y)
    return trained_data

# Estimate the battery life
def estimate_battery_life(charge_time):
    if charge_time < 4:
        battery_life = 2.00 * charge_time
    else:
        battery_life = 8
    return battery_life

if __name__ == '__main__':
    trained_data = train_classifier()
    timeCharged = float(input().strip())
    prediction = trained_data.predict(timeCharged)
    predicted = prediction[0][0]
    #print(predicted) --> The predicted output is always 2 * timeCharged, if timeCharged < 4, else 8

    # Hence we estimate battery life using the estimate_battery_life() function
    battery_life = estimate_battery_life(timeCharged)
    print(battery_life)


