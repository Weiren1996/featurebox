import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from featurebox.selection.backforward import BackForward
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

warnings.filterwarnings("ignore")

# 数据导入
store = Store(r'C:\Users\Administrator\Desktop\zlj')
data = Call(r'C:\Users\Administrator\Desktop\zlj', index_col=None)
all_import = data.csv().zlj

x_name = all_import.index.values
y = all_import["y"].values
x = all_import.iloc[:, 1:].values

# # 预处理
minmax = MinMaxScaler()
x = minmax.fit_transform(x)
# 数据划分
xtrain, xtest = x[20:], x[:20]
ytrain, ytest = y[20:], y[:20]

xtrain, ytrain = sklearn.utils.shuffle(xtrain, ytrain)

# x = minmax.inverse_transform(x_new)

# # 网格搜索*前进后退


me4 = svm.SVR(kernel='rbf', gamma='auto', degree=3, tol=1e-3, epsilon=0.1, shrinking=True, max_iter=2000)
# 网格
param_grid4 = [{'C': [10000, 100, 50, 10, 1, 0.5, 0.1], "epsilon": [1, 0.1, 0.01, 0.001, 0.0001]}]
gd = GridSearchCV(me4, cv=10, param_grid=param_grid4, scoring='neg_mean_absolute_error', n_jobs=1)
# 前进后退
ba = BackForward(gd, n_type_feature_to_select=6, primary_feature=None, muti_grade=2, muti_index=None,
                 must_index=None, tolerant=0.01, verbose=0, random_state=0)
x_add = np.concatenate((x, xtest), axis=0)
y_add = np.concatenate((y, ytest), axis=0)
# running!
ba.fit(x_add, y_add)
xtest = xtest[:, ba.support_]
xtrain = xtrain[:, ba.support_]

# 预测
scoretest = ba.estimator_.score(xtest, ytest)
scoretrain = ba.estimator_.score(xtrain, ytrain)
y_pre_test = ba.estimator_.predict(xtest)
y_pre_train = ba.estimator_.predict(xtrain)

# 训练#
cor_ = abs(y_pre_train - ytrain) / ytrain
cors_ = cor_.mean()
# 测试

cor = abs(y_pre_test - ytest) / ytest
cors = cor.mean()
# 合并
y_ = ba.estimator_.predict(x[:, ba.support_])

all_import["y_predict"] = y_


# 画图

def scatter(y_true, y_predict, strx='y_true', stry='y_predict'):
    x, y = y_true, y_predict
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, marker='o', s=50, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
    ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
    plt.xlabel(strx)
    plt.ylabel(stry)


scatter(ytest, y_pre_test, strx='y_test(GWh)', stry='y_predict(Gwh)')
scatter(ytrain, y_pre_train, strx='y_train(GWh)', stry='y_predict(GWh)')


def scatter2(x, y_true, y_predict, strx='y_true', stry1='y_true(GWh)', stry2='y_predict', stry="y"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    l1 = ax.scatter(x, y_true, marker='o', s=50, alpha=0.7, c='orange', linewidths=None, edgecolors='blue')
    ax.plot(x, y_true, '-', ms=5, lw=2, alpha=0.7, color='black')
    l2 = ax.scatter(x, y_predict, marker='^', s=50, alpha=0.7, c='green', linewidths=None, edgecolors='blue')
    ax.plot(x, y_predict, '-', ms=5, lw=2, alpha=0.7, color='green')
    # ax.plot([min(x), max(x)], [min(x), max(x)], '--', ms=5, lw=2, alpha=0.7, color='black')
    plt.xlabel(strx)
    plt.legend((l1, l2),
               (stry1, stry2),
               loc='upper left')
    plt.ylabel(stry)


a = np.arange(1, 185)
scatter2(np.arange(1, 185), y[::-1], y_[::-1], strx='month', stry="y(Gwh)", stry1='y_true(GWh)', stry2='y_predict(GWh)')

# #导出

store.to_pkl_sk(ba.estimator_, "model")
store.to_csv(all_import, "predict")
print(all_import.iloc[:, 1:-1].columns.values[ba.support_])
