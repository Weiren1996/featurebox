# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from featurebox.selection.quickmethod import method_pack
from featurebox.tools.exports import Store

# 数据导入

store = Store(r'/data/home/wangchangxin/data/wr/pvc')

# """for element site"""
data = pd.read_excel(r'/data/home/wangchangxin/data/wr//pvc/wrpvc.xlsx',
                     header=0, skiprows=None, index_col=0)

y = data["t"].values
x_p_name = ['t_t', 'v', 'b', 'hat', 'd']
x = data[x_p_name].values

# # # 预处理
minmax = MinMaxScaler()
x = minmax.fit_transform(x)

# x_, y_ = shuffle(x, y, random_state=2)
# x_, y_ = x,y

# # 建模
method_all = ['SVR-set', "GPR-set", "RFR-em", "AdaBR-em", "DTR-em", "LASSO-L1", "BRR-L1"]
methods = method_pack(method_all=method_all,
                      me="reg", gd=True)
pre_y = []
ests = []
for name, methodi in zip(method_all, methods):
    if name in ['SVR-set', "LASSO-L1", "BRR-L1"]:
        cv = LeaveOneOut()
    else:
        cv = 5
    methodi.cv = cv
    methodi.scoring = "neg_root_mean_squared_error"
    gd = methodi.fit(X=x, y=y)
    # score = gd.best_score_
    est = gd.best_estimator_
    score = cross_val_score(est, X=x, y=y, scoring="neg_root_mean_squared_error", cv=cv).mean()
    print(name, "neg_root_mean_squared_error", score)
    pre_yi = cross_val_predict(est, X=x, y=y, cv=cv)
    pre_y.append(pre_yi)
    ests.append(est)
    store.to_pkl_pd(est, name)

pre_y.append(y)
pre_y = np.array(pre_y).T
pre_y = pd.DataFrame(pre_y)
pre_y.columns = method_all + ["realy_y"]
store.to_csv(pre_y, "预测值")
