import time
import numpy as np
from numpy.linalg import svd
from cur import cur_decomposition
from dataset import *
import math
C,U,R=cur_decomposition(full_data_set,900)
a=np.dot(C,U)
b=np.dot(a,R)
#print(b)
b=b-full_data_set
b=np.square(b)
b=np.sum(b)
b=math.sqrt(b)
print(b/10000000000000000)
#print(np.dot(np.dot(C,U),R))
