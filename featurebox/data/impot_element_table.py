import pandas as pd
import os
pa = os.path.abspath(__file__)
pa = os.path.split(pa)[0]
pa = os.path.join(pa, r"element_table.xlsx")

element_table = pd.read_excel(pa,
                              header=7, skiprows=0, index_col=0)
