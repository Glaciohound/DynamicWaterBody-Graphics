#!/usr/bin/env python
# coding=utf-8

import os

series = 13
area = [(),
        (700, 1200, 100, 750),
        (200, 600, 300, 600),
        (100, 200, 100, 200),
        (100, 300, 160, 220),
        (100, 800, 200, 600),
        (200, 500, 200, 500),
        (200, 500, 300, 600),]

for i in range(1, 8):
    print("./main data/{0}.bmp {1}{0}.bmp {2} {3} {4} {5}".format(i, series, area[i][0], area[i][1], area[i][2], area[i][3]))
    os.system("./main data/{0}.bmp {1}{0}.bmp {2} {3} {4} {5}".format(i, series, area[i][0], area[i][1], area[i][2], area[i][3]))
