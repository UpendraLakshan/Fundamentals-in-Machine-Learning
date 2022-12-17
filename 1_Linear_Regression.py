import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {'X':[1,2,3,4,5,6,7,8,9,10],
        'y':[10,20,30,40,50,60,70,80,90,100]}
df = pd.DataFrame(data)

X = df["X"]
y = df["y"]
plt.scatter(X, y,color='red',marker='+')
plt.show()

X = np.array(X).reshape(-1,1)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size = 0.1, random_state = 0)


LinearRegressionModel= LinearRegression()
LinearRegressionModel.fit(X_Train, Y_Train)

print(X_Test)
y_pred = LinearRegressionModel.predict(X_Test)   
print('pred',y_pred)

#Test with your own data
Test=[11]
Test= np.array(Test).reshape(-1,1)
y_pred = LinearRegressionModel.predict(Test)

print('Prediction',y_pred)
