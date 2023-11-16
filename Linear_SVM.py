import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1,6.4,7.9,2.5])
y = np.array([3, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5,7.4,5.9,11.5])

# show unclassified data
#plt.scatter(X, y, c = y, edgecolors='red')
plt.scatter(X, y, c = y, alpha= 1, edgecolors='red')
plt.show()

# shaping data for training the model
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]

# define the model
clf = svm.SVC(kernel='linear', C = 1.0)

# train the model
clf.fit(training_X, training_y)

# get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(0, 13)

# get the y-values to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y, edgecolors='red')
plt.legend()
plt.show()






