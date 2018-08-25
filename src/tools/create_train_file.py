from os import listdir
from os.path import isfile, join

train_file = open("/home/trunghieu11/Work/voice_zaloai/src/labels/test_gender.csv", "a+")

my_path = ["/home/trunghieu11/Work/voice_zaloai/data/public_test"]

for path in my_path:
    for f in listdir(path):
        if isfile(join(path, f)):
            train_file.write(f + "\n")

train_file.close()