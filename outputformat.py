# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:35:32 2020

@author: mk909
"""

import numpy as np

data_tags = [x.split() for x in data_text_arr]

res = [0]*len(data_text_arr)
for i in range(0, len(data_text_arr)):
    res[i] = list(zip(data_tags[i], tag_pred[i]))
    

y = []
x = PrettyTable()

x.field_names = slots

for i in range(0, len(tag_pred)):
    z = []
    l = tag_pred[i]
    for j in slots:
        k = j
        if j in l:
            z.append(j)
        else:
            z.append(0)
    y.append(z)


result = []           
for i in range(0,len(res)):
    first = res[i]
    second = y[i]
    third = y[i]
    fourth = np.array(first)
    for i in range(0,len(first)):
        for j in range(0,len(second)):
            if fourth[i][1] == second[j]:
                third[j] = fourth[i][0]
    result.append(third)

for i in range(len(result)):
    x.add_row(result[i])