########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.


# %matplotlib inline

# ipynb needs  to be modified.
import csv
import numpy as np

# Size taken from the dataset and is known
X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

data_filename = "./ionosphere.data"

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # print (i, row[-1], row[:-1]) # for example, 318 g ['1', '0', '0.74790', '0.00840', '0.83312', '0.01659', '0.82638', '0.02469', '0.86555', '0.01681', '0.60504', '0.05882', '0.79093', '0.04731', '0.77441', '0.05407', '0.64706', '0.19328', '0.84034', '0.04202', '0.71285', '0.07122', '0.68895', '0.07577', '0.66387', '0.08403', '0.63728', '0.08296', '0.61345', '0.01681', '0.58187', '0.08757', '0.55330', '0.08891', 'g'], all in character!
        # Get the data, converting each item to a float
        X[i] = [float(datum) for datum in row[
                                          :-1]]  # 從頭到最後的前一個(Attention to the difference between row[-1] and row[:-1]) list comprehension (單行for敘述)
        # Set the appropriate row in our dataset
        # X[i] = data
        # 1 (True) if the class is 'g', 0 (False) otherwise
        y[i] = row[-1] == 'g'

# import pandas as pd
# iono = pd.read_csv("./data/ionosphere.data", header=None)
#
# X = iono.iloc[:, :-1]
# y = iono.iloc[:, -1]

X
X.shape
# help(X.std)
# X.std(axis=0) # So, that's why we skip scaling.
y
y.shape

### Data types and missing values identification
X.dtype
# X.isnull.sum()

# {0:.2f}.format()
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))

print("The class distribution of training set is\n{}.".format(y.value_counts() / len(y)))

print("The class distribution of training set is\n{}.".format(y_train.value_counts() / len(y_train)))

print("The class distribution of test set is\n{}.".format(y_test.value_counts() / len(y_test)))

### Standardization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_std = sc.fit_transform(X)  # for cross-validation

from sklearn.neighbors import KNeighborsClassifier  # import module

estimator = KNeighborsClassifier()  # model spec., use default settings

estimator.fit(X_train_std, y_train)  # Minkowski dist. p = 2, model training

# dir(estimator)
# estimator.get_params()
# for name in ['metric','n_neighbors','p']:
#    print(estimator.get_params()[name])

# help(KNeighborsClassifier)
train_pred = estimator.predict(X_train_std)
train_pred[:5]
y_train[:5]
import numpy as np

train_acc = np.mean(y_train == train_pred) * 100
print("The accuracy of training set is {0:.1f}%".format(train_acc))

y_pred = estimator.predict(X_test_std)  # model prediction
y_pred[:5]
y_test[:5]
test_acc = np.mean(y_test == y_pred) * 100
print("The accuracy of test set is {0:.1f}%".format(test_acc))

from sklearn.model_selection import cross_val_score

# 交叉驗證一次#default 3-cv
scores = cross_val_score(estimator, X_std, y, scoring='accuracy')
# scores.shape # (3,)
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))

# help(cross_val_score)


avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    scores = cross_val_score(estimator, X_std, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

# type(avg_scores) # "list"

len(avg_scores)  # 20

# type(all_scores) # "list"

# len(all_scores) # 20

all_scores  # 3-fold cross validation

### plotting avg_scores under different k
from matplotlib import pyplot as plt

# plt.figure(figsize=(32,20))
# plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
# plt.axis([0, max(parameter_values), 0, 1.0])

# from matplotlib import ticker
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xticks(np.arange(0, 21))
plt.xlabel('No. of nearest neighbours')
plt.ylabel('Average accuracy under 3-CV')
ax.plot(parameter_values, avg_scores, '-o')
# xticks = parameter_values
# ticklabels = np.array(parameter_values).astype('str').tolist()
# ax.xticks(xticks, ticklabels)
# plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())

# fig.savefig('./data/iono_tuning_avg_scores.png', bbox_inches='tight')
# fig.savefig('./data/iono_tuning_avg_scores.png')


plt.plot(parameter_values, avg_scores, '-o')

### plotting all_scores of three folds CV under different k
for parameter, scores in zip(parameter_values, all_scores):
    n_scores = len(scores)  # 3 folds
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    ax.plot([parameter] * n_scores, scores, '-o')
    plt.xticks(np.arange(0, 21))
    plt.xlabel('No. of nearest neighbours')
    plt.ylabel('Accuracy of 3-CV')
    plt.plot([parameter] * n_scores, scores, '-o')  # parameters放長三倍

# plt.savefig('./data/iono_tuning_all_scores.png')

list(zip(parameter_values, all_scores))

plt.plot(parameter_values, all_scores, 'bx')  # 另一種繪法！

from collections import defaultdict

all_scores = defaultdict(list)  # 用dict存放scores
print(all_scores)
parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    for i in range(100):  # 重複100次的十摺cv
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X_std, y, scoring='accuracy', cv=10)
        all_scores[n_neighbors].append(scores)

for parameter in parameter_values:
    scores = all_scores[parameter]
    n_scores = len(scores)  # 重複100次的十摺cv結果
    # print(n_scores) # 100
    # print(scores)
    plt.xticks(np.arange(0, 21))
    plt.xlabel('No. of nearest neighbours')
    plt.ylabel('Accuracy of repeated 10-CV')
    plt.plot([parameter] * n_scores, scores, '-o')

type(all_scores)

all_scores.keys()

type(all_scores[1])

len(all_scores[1])  # 不同k值下，重複進行一百次的十摺交叉驗證

type(all_scores[1][0])

all_scores[1][0]  # k = 1 時，第一次的十摺交叉驗證結果

all_scores[1][99]  # k = 1 時，第一百次的十摺交叉驗證結果

# 另一種尺度調整方式
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

scaling_predicting_pipeline = Pipeline([('scale', MinMaxScaler()), ('predict', KNeighborsClassifier())])
# scaling_predicting_pipeline
# type(scaling_predicting_pipeline)

# 丟X就可以了！
scores = cross_val_score(scaling_predicting_pipeline, X, y, scoring='accuracy')

scores

print("The pipeline scored an average accuracy for is {0:.1f}%".format(np.mean(scores) * 100))
