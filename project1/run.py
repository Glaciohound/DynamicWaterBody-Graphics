#!/usr/bin/env python
# coding=utf-8

import os

os.system("make")
#range_ = range(50, 70)
#range_ = range(70, 90)
range_ = range(90, 110)
#range_ = range(110, 130)
#range_ = range(130, 150)
not_range = [0, 1, 2, 3,
             30, 31, 32, 33,
             60, 61, 62, 63,
             90, 91, 92, 93,
             120, 121, 122, 123]

for i in range_:
    if not i in not_range:
        print(i)
        os.system("./main output " + str(i) + " 100000000")
