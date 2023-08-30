from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import cross_validate


lr_ = [0.00001, 0.0001, 0.001, 0.01, 0.1]
architecture = {'0 hidden layer': (),
                '1 hidden layer with 2 nodes': (2,),
                '1 hidden layer with 6 nodes': (6,),
                '2 hidden layers with 2 and 3 nodes': (2, 3),
                '2 hidden layers with 3 and 2 nodes': (3, 2),
                '2 hidden layers with 8 and 4 nodes': (8, 4),
                '3 hidden layers with 4 nodes each': (4, 4, 4)}

acc = dict()

for foo in architecture.keys():
    cur_acc = dict()
    for lr in lr_:
        clf = MLP(hidden_layer_sizes=architecture[foo], solver='adam', learning_rate_init=lr, max_iter=3000)
        cv_results = cross_validate(clf, X, y, cv=5)
        score = np.mean(cv_results['test_score'])
        cur_acc[lr] = score
        print(score)
    acc[foo] = cur_acc

result = pd.DataFrame(acc)

plt.figure(dpi=400)
plt.xlabel('Learning rate η')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.grid()
for foo in architecture.keys():
    plt.plot(result[foo], label=foo)
plt.title('Accuracy using simpler models')
plt.legend()
plt.show()


def ANN(x_train, y_train, x_val, y_val, x_test, y_test, lr, hidden_layer_sizes, max_iter, activation, solver, random_state):
    clf = MLP(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, activation=activation, solver=solver, random_state=random_state)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    return acc, kappa


from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold as RSKF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.ensemble import StackingClassifier as SC
from sklearn.svm import SVC
from sklearn import preprocessing
# Pre-processing the data

conn_est = ConnectivityMeasure(kind='correlation') # Connectivity Estimator
conn_matrices = conn_est.fit_transform(data['rois_aal'])

X = sym_to_vec(conn_matrices) # Converting sym. matrix into vector

y = data.phenotypic['DX_GROUP']
y[y == 2] = -1


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
def get_stacking():
    level0 = list()
    level0.append(('lr', LR()))
    level0.append(('knn', KNN(n_neighbors=29, p=1)))
    level0.append(('svm', SVC(gamma=0.000122, C=4)))

    level1 = LR()

    model = SC(estimators=level0, final_estimator=level1, cv=5)
    return model


def get_models():
    models = dict()
    models['lr'] = LR()
    models['knn'] = KNN(n_neighbors=29, p=1)
    models['svm'] = SVC(gamma=0.000122, C=4)
    models['stacking'] = get_stacking()
    return models


def evaluate_model(clf, X, y):
    cv_results = cross_validate(clf, X, y, cv=5)
    return cv_results['test_score']


models = get_models()
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
plt.figure(dpi=400)
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Box Plot of Standalone and Stacking Model Accuracies for Binary Classification')
plt.show()

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier as KNN

n_neighbors = list(range(1, 51))


# First, we go with Manhattan distance (p = 1)

acc = []
K = []
for k in range(1, 51):
    K.append(k)
    clf = KNN(n_neighbors=k, p=1)
    cv_results = cross_validate(clf, X, y, cv=5)
    score = np.mean(cv_results['test_score'])
    acc.append(score)

# First, we go with Euclidean distance (p = 2)

acc_ = []
K_ = []
for k in range(1, 51):
    K_.append(k)
    clf = KNN(n_neighbors=k, p=2)
    cv_results = cross_validate(clf, X, y, cv=5)
    score = np.mean(cv_results['test_score'])
    acc_.append(score)

# Visualizing the results obtained
plt.figure(dpi = 400)
plt.xlabel('Value of K in KNN')
plt.ylabel('Accuracy observed')
plt.grid()
plt.plot(K, acc, label='Manhattan')
plt.plot(K_, acc_, label='Euclidean')
plt.title('Accuracy observed for KNN')
plt.legend()
plt.show()