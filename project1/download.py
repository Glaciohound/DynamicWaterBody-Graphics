#!/usr/bin/env python
# coding=utf-8

import os

dataset = {
    "aws1": [49],
    #"aws2": [147, 148],
    #"aws3": [149, 150],
    #"vultr3": [100],
}
for server in dataset.keys():
    for i in dataset[server]:
        if not os.path.exists("../../outputs/output{}.bmp".format(i)):
            if server[0] == 'a':
                print('scp -i ~/Downloads/aws_personal.pem {0}:~/source/project1/outputs/output{1}.bmp ../../outputs/'.format(server, i))
                os.system('scp -i ~/Downloads/aws_personal.pem {0}:~/source/project1/outputs/output{1}.bmp ../../outputs/'.format(server, i))
            else:
                print('scp vultr3:~/source/project1/outputs/output{0}.bmp ../../outputs/'.format(i))
                os.system('scp vultr3:~/source/project1/outputs/output{0}.bmp ../../outputs/'.format(i))
