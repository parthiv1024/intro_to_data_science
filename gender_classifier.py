from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf1 = tree.DecisionTreeClassifier()
clf2 = svm.SVC()
clf3 = GaussianNB()
clf4 = MLPClassifier(hidden_layer_sizes=(5, 2))

clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)
clf4 = clf4.fit(X, Y)

print("Decision Tree Classifier: " + str(accuracy_score(Y, clf1.predict(X))))
print("Support Vector Machine Classifier: " + str(accuracy_score(Y, clf2.predict(X))))
print("Naive Bayes Classifier: " + str(accuracy_score(Y, clf3.predict(X))))
print("Neural Network Classifier: " + str(accuracy_score(Y, clf4.predict(X))))
