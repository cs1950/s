import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import metrics

df=pd.read_csv('C:/Users/rushi/Desktop/diabetes.csv')

df.columns

df.isnull().sum()

df.shape
X = df.drop('Outcome',axis = 1)
y = df['Outcome']


from sklearn.preprocessing import scale
X = scale(X)
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Prediction: ",y_pred)

print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_test,y_pred)
print(cs)

print("Acccuracy ",metrics.accuracy_score(y_test,y_pred)*100)

total_misclassified = cs[0,1] + cs[1,0]
print(total_misclassified)
total_examples = cs[0,0]+cs[0,1]+cs[1,0]+cs[1,1]
print(total_examples)
print("Error rate",total_misclassified/total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred))

print("Precision score",metrics.precision_score(y_test,y_pred))

print("Classification report ",metrics.classification_report(y_test,y_pred))

from sklearn.svm import SVC

# cost C = 1
model_svc = SVC(C = 1)

#Train the Model using fit Function
model_svc.fit(X_train, y_train)

#Predict the Trained Model
y_pred_svc = model_svc.predict(X_test)

print("Confusion matrix: ")
cs1 = metrics.confusion_matrix(y_test,y_pred_svc)
print(cs1)

print("Acccuracy ",metrics.accuracy_score(y_test,y_pred_svc))

total_misclassified = cs1[0,1] + cs1[1,0]
print(total_misclassified)
total_examples = cs1[0,0]+cs1[0,1]+cs1[1,0]+cs1[1,1]
print(total_examples)
print("Error rate",total_misclassified/total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred_svc))

print("Precision score",metrics.precision_score(y_test,y_pred_svc))

print("Classification report ",metrics.classification_report(y_test,y_pred_svc))
