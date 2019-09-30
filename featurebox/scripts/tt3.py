import numpy as np
# from numpy import random
from featurebox.scripts.ttt import rr2

import random

def rr():
    return rr2()

def prr(random_state=0):
    random.seed(random_state)
    return rr()