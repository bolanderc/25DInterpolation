#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:57:09 2019

@author: christian
"""

import numpy as np

sectiony = -0.62767463922501

data = np.genfromtxt('wing_lower.csv', skip_header=1, delimiter=',')[:, 1:4]
sep = np.where(data[:, 1] >= sectiony)
data_0 = data[sep, :]
print(data_0)
sep = np.where(data[:, 1] < sectiony)
data_1 = data[sep, :]
print(data_1)
np.savetxt('wing_lower_0', data_0[0])
np.savetxt('wing_lower_1', data_1[0])
print(data[np.argmin(data[:, 1]), :])
slic = np.where(data[:, 1] <= -0.9015)
print(data[slic, :])
np.savetxt('wing_edge_2_l', data[slic, :][0])
