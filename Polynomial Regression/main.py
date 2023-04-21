import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics

data = pd.read_csv('SuperMarketSales.csv')
############## preprocessing ###################################################
# Drop the rows that contain missing values
data.dropna(how='any', inplace=True)

# show correlation between features and Weekly_Sales
print(data.corr())

# best 2 correlation are Store and CPI
X = data[['Store', 'CPI']]
Y = data["Weekly_Sales"]

# split Data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.44, random_state=10)

# scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


########### functions ###############
def degree_2(X):
    out = pd.DataFrame()
    c = 0
    for i in range(0, 2):
        for j in range(i, 2):
            out[''.join(sorted(f'{X.columns[i]}*{X.columns[j]}'))] = X.iloc[:, i] * X.iloc[:, j]
            c += 1
    return out


def degree_func(degree, X):
    c = 0
    output = pd.DataFrame(X)
    X = pd.DataFrame(X)
    temp_df = pd.DataFrame(degree_2(X))
    for i in range(2, degree):  # 2
        m = len(temp_df.columns)  # 3
        for j in range(len(X.columns)):  # 1 b
            for k in range(c, m):  # 0 b2
                temp_df[''.join(sorted(f'{X.columns[j]}*{temp_df.columns[k]}'))] = X.iloc[:, j] * temp_df.iloc[:,
                                                                                                  k]  # 0,3 -> 4,9 -> ,14
        c += len(temp_df.columns) - m + 1
    output = pd.concat([output, temp_df], axis=1)
    output = output.T.drop_duplicates().T
    return output


def gradient_descent(y_train, X_train):
    L = 0.01  # The learning Rate
    epochs = 10000  # The number of iterations to perform gradient descent

    n = float(len(X_train))  # Number of elements in X
    X_train = np.array(X_train)
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    y_train = y_train[:, np.newaxis]
    theta = np.zeros((X_train.shape[1], 1))

    # Train the model
    for i in range(epochs):
        y_pred = np.dot(X_train, theta)
        deri = -(2 / n) * np.dot(X_train.T, y_train - y_pred)
        theta = theta - L * deri
    return theta


def addColumnOfOnes(X):
    X = np.array(X)
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X


################### model ###################################

for i in range(2, 10):
    x_train = degree_func(i, X_train)
    x_test = degree_func(i, X_test)
    theta = gradient_descent(y_train, x_train)
    # Add a column of ones to x_test and x_train
    x_test = addColumnOfOnes(x_test)
    x_train = addColumnOfOnes(x_train)
    # Make predictions on train and test data
    prediction_test = np.dot(x_test, theta)
    prediction_train = np.dot(x_train, theta)
    # Calculate mean squared error for test data
    print(f'Mean Square Error of train of degree {i} = ',
          metrics.mean_squared_error(np.asarray(y_train), prediction_train))
    print(f'Mean Square Error of test of degree {i} = ',
          metrics.mean_squared_error(np.asarray(y_test), prediction_test))