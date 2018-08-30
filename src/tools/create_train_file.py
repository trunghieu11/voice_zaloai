from os import listdir
from os.path import isfile, join

train_file = open("../labels/train_accent.csv", "a+")

list_accent = ["north", "central", "south"]

for i, accent in enumerate(list_accent):
    my_path = ["../../data/train/female_" + accent, "../../data/train/male_" + accent]

    for path in my_path:
        for f in listdir(path):
            if isfile(join(path, f)):
                train_file.write(f + "," + str(i) + "\n")

train_file.close()