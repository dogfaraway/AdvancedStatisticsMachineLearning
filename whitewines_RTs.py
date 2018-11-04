############################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
##############################################
### Data sets: germancredit.csv
### Notes: This code is provided without warranty.

import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
wine = pd.read_csv("./whitewines.csv")
wine.dtypes
wine.describe(include='all')
ax = wine.quality.hist()
fig = ax.get_figure()
fig.savefig("./quality_hist.png")

# ![葡萄酒評點分數直方圖](./_img/pd_boxplot.png)

### Introduction to Model Trees from scratch (https://towardsdatascience.com/introduction-to-model-trees-6e396259379a)
# Decision tree can in principal take on any model during the tree splitting procedure i.e. linear regression, logistic regression, neural networks. 

### Data Splitting
X = wine.drop(['quality'], axis=1)
y = wine['quality']
X_train = X[:3750]
X_test = X[3750:]

y_train = y[:3750]
y_test = y[3750:]

from sklearn import tree

clf = tree.DecisionTreeRegressor()  # 21xx nodes
clf.get_params()
# {'criterion': 'mse',
# 'max_depth': None,
# 'max_features': None,
# 'max_leaf_nodes': None,
# 'min_impurity_decrease': 0.0,
# 'min_impurity_split': None,
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'min_weight_fraction_leaf': 0.0,
# 'presort': False,
# 'random_state': None,
# 'splitter': 'best'}

clf = clf.fit(X_train, y_train)

### Check Overfitting or not
n_nodes = clf.tree_.node_count  # 21xx nodes
print('The tree has {0} nodes.'.format(n_nodes))

### Model Improvement
# clf = tree.DecisionTreeRegressor(max_leaf_nodes = 10, min_samples_leaf = 5, max_depth= 5) # 19 nodes
# clf = tree.DecisionTreeRegressor(max_leaf_nodes = 8, min_samples_leaf = 7, max_depth= 30) # 15 nodes, MSE_test < MSE_train
clf = tree.DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=7,
                                 max_depth=30)  # 19 nodes by similar settings to R
# clf = tree.DecisionTreeRegressor(min_samples_leaf = 7, max_depth= 30) # 827 nodes

clf = clf.fit(X_train, y_train)

### Check Overfitting or not
n_nodes = clf.tree_.node_count
print('The tree has {0} nodes.'.format(n_nodes))

### Model Performance
y_train_pred = clf.predict(X_train)

y_train.describe()
pd.Series(y_train_pred).describe()  # shrinked distribution

y_test_pred = clf.predict(X_test)
y_test.describe()
pd.Series(y_test_pred).describe()  # shrinked distribution

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)))

### Decision tree report
n_nodes = clf.tree_.node_count
print('The tree has {0} nodes.'.format(n_nodes))
children_left = clf.tree_.children_left
print('The left children for each node are {0}, respectively.'.format(children_left))
children_right = clf.tree_.children_right
print('The right children for each node are {0}, respectively.'.format(children_right))

# print('The right children for each node are', '\n','{0}, respectively.'.format(children_right))

feature = clf.tree_.feature
print('The indices of splitting features employed in each node are {0}, respectively.'.format(feature))
threshold = clf.tree_.threshold
print('The correponding thresholds for each splitting features are {0}, respectively.'.format(threshold))

# Drawing by `node_depth`, `is_leaves` and `stack`
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)  # default to False
stack = [(0, -1)]  # seed is the root node id and its parent depth (node_id, parent_depth)
while len(stack) > 0:  # stack最後為空
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1  # 自己的深度為父節點深度加1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))  # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_id], parent_depth + 1))  # 加右分枝節點，分枝節點的父節點深度正是自己的深度
    else:
        is_leaves[node_id] = True  # is_leaves預設為False，最後有True有False

print("各節點的深度分別為：{0}".format(node_depth))
print("各節點是否為終端節點的真假值分別為：{0}".format(is_leaves))
print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print(
            "%snode=%s leaf node." % (node_depth[i] * "\t", i))  # 注意node_depth[i] * "\t"是依各節點深度縮進，對應第1個%s(try 2 * "\t")
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
print()

### Tree plotting
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import \
    pydot  # 注意Spyder在Mac中的套件安裝過程： from /usr/local/lib/python3.5/site-packages/ to Spyder.app/Contents/Resources/lib/python3.5 (pip install pydot可)
import pydotplus

# What is StringIO in python used for in reality? https://stackoverflow.com/questions/7996479/what-is-stringio-in-python-used-for-in-reality
dot_data = StringIO()  # 樹狀模型先output到dot_data
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                    'alcohol'])
# (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())

# dir(dot_data) # 確定有getvalue方法

# Need Graphviz (http://www.graphviz.org/) Download graphviz-2.36.0.pkg for Mac OS Mountainlion
# Please install graphviz-2.38.msi Windows
# Add D:\Graphviz2.38\bin to Path in Control Panel (以系統管理者啟動MS-DOS，再安裝到D:\)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  # 從dot_data產生graph
graph.write_pdf("credit.pdf")  # 寫出graph

graph  # <pydotplus.graphviz.Dot at 0xa1b30cdd8>

graph.write_png('whitewines.png')
from IPython.core.display import Image

Image(filename='whitewines.png')
