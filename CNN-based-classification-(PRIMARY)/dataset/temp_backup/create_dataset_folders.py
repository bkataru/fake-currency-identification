import os

cur_path = '.\\'
old_exceptions = ['200', '500', '2000']
types = ['old', 'new']
sides = ['front', 'back']
orientations = ['up', 'down']
folders = ['10', '20', '50', '100', '200', '500', '2000']

for fold in folders:
    for ty in types:
        if ty == 'old' and fold in old_exceptions:
            continue
        for sid in sides:
            for ori in orientations:
                os.mkdir(cur_path + 'training_set\\' + "{}_{}_{}_{}".format(fold, ty, sid, ori))

