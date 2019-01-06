# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:20:08 2014

@author: bmmorris
"""

import numpy as np
a = open('variablepixels.csv').readlines()
img = []
for line in a:
    img.append(map(float,line.split(',')))
img = np.array(img)
np.save('variablepixels.npy',img)

from matplotlib import pyplot as plt
plt.imshow(img)
plt.show()
