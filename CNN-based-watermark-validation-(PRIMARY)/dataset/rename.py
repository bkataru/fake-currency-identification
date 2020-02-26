import os

for filename in os.listdir():
    if not filename.split('.')[1] == 'py':
        name = filename.split('.')[0]
        os.rename(filename, '{}-pr.jpg'.format(name))