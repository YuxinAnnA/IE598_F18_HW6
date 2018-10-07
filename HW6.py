from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# load data and assign them to X and y
iris = load_iris()
X, y = iris.data, iris.target

# part 1 (training data)
random_state = range(1, 11)
scores = []
for r in random_state:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=r)
    entropy = DecisionTreeClassifier(criterion="entropy", random_state=r, max_depth=3)
    entropy.fit(X_train, y_train)
    y_pred_train = entropy.predict(X_train)
    scores.append(metrics.accuracy_score(y_train, y_pred_train))
plt.plot(random_state, scores, 'ro')
plt.title('Decision Tree for training dataset: Varying random state')
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.show()
print np.mean(scores)
print np.std(scores)

# part 1 (testing data)
random_state = range(1, 11)
scores = []
for r in random_state:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=r)
    entropy = DecisionTreeClassifier(criterion="entropy", random_state=r, max_depth=3)
    entropy.fit(X_train, y_train)
    y_pred_test = entropy.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred_test))
plt.plot(random_state, scores, 'ro')
plt.title('Decision Tree for test dataset: Varying random state')
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.show()
print np.mean(scores)
print np.std(scores)



# part 2
random_state = range(1, 11)
scores = []
for r in random_state:
    entropy = DecisionTreeClassifier(criterion="entropy", random_state=r, max_depth=3)
    scores = cross_val_score(estimator=entropy, X=X_train, y=y_train, cv=10, n_jobs=1)
plt.plot(random_state, scores, 'ro')
plt.title('Decision Tree cross validation training: Varying random state')
plt.xlabel('Random state')
plt.ylabel('Accuracy')
plt.show()
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.4f +/- %.4f' % (np.mean(scores),np.std(scores)))

### Name
print("My name is Yuxin Sun")
print("My NetID is: yuxins5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
