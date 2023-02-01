from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

bc=load_breast_cancer()
#print(bc)
X=scale(bc.data)

y=bc.target 
#print(X)
#print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=KMeans(n_clusters=2,random_state=0)

model.fit(X_train)

predictions=model.predict(X_test)

labels=model.labels_

print('Label',labels)
print('predictios',predictions)
print('acc',accuracy_score(y_test,predictions))
print('actual',y_test)
print(pd.crosstab(y_train,labels))