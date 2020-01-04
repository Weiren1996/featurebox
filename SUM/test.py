# -*- coding: utf-8 -*-

# @Time    : 2019/12/20 15:11
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from featurebox.tools.exports import Store
from featurebox.tools.imports import Call

store = Store(r'C:\Users\Administrator\Desktop\band_gap_exp\4.symbol', )
data = Call(r'C:\Users\Administrator\Desktop\band_gap_exp\4.symbol')
store.to_csv(data.filename)
