import json
from collections import defaultdict
import matplotlib.pyplot as plt
# Load the dataset annotations
with open('train_b-10classes.json', 'r') as f:
    dataset = json.load(f)

print(len(dataset))




new_data = []
# Convert class labels
for data in dataset:
    new_dict = {'id':data['id'], 'template':data['template']}    
    new_data.append(new_dict)

print(new_data, len(new_data))

print(new_data, len(new_data))
# Save the modified dataset
with open('train_ann_b-10classes.json', 'w') as f:
    json.dump(new_data, f)