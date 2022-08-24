#
# Yolo Data Splitter
# Developed by Scott Uneberg, https://github.com/robotwhispering
# Under MIT License
# Updated 09/04/2020
#

from pathlib import Path

# set path to /darknet/data directory (subfolders will be recursively included)
img_path = "./data"

# percentage of images to be used for the test set
pcnt_test = 0
r = range(10,30+1)
while not pcnt_test:
    try:
        pcnt_test = int(input("Enter the percentage for testing (10-30): "))
        if pcnt_test not in r:
            raise ValueError
    except ValueError:
        pcnt_test = 0
        print("Please enter an integer between 10 and 30!")

# create and/or truncate train.txt and test.txt
file_train = open("train.txt", "w")
file_test = open("test.txt", "w")

# populate train.txt and test.txt
iterator = 1
test_cnt = 0
train_cnt = 0
index_test = round(100 / pcnt_test)

for path in Path(img_path).rglob("*.jpg"):

    if iterator == index_test:
        iterator = 1
        file_test.write(str(path.resolve()) + "\n")
        test_cnt += 1
    else:
        file_train.write(str(path.resolve()) + "\n")
        iterator += 1
        train_cnt += 1

# display image path on screen
print("Image directory: " + str(Path(img_path).resolve()))

# display results to console
print("Training/Testing percentage split: " + str(pcnt_test))
print("Training set: " + str(train_cnt) + " files")
print("Testing set: " + str(test_cnt) + " files")
print("Dataset splitting complete!")