makeSubsets.py
in train.json
out train-10classes.json


makeBalanced.py:
input 'train-10classes.json'
output train_b-10classes.json

getOnly2Columns.py
in train_b-10classes.json
out train_ann_b-10classes.json


# Test set
makeTestSubsets.py
in test-answers.csv
out test-answers-10classes.csv