import os

for block in range(3, 30, 2): # > 0
    for constant in range(-7, 7):
        os.mkdir('compiled\\{}-{}'.format(block, constant))