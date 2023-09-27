import json
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def sample_thousand_rows(input_list):
    if len(input_list) <= 1000:
        return random.sample(input_list, len(input_list))
    else:
        return random.sample(input_list, 1000)

# Load the dataset annotations
with open('train-10classes.json', 'r') as f:
    dataset = json.load(f)

print(len(dataset))

# Mapping of existing class labels to new class labels
class_numbers = defaultdict(list)

# new_data = []
# Convert class labels
for data in dataset:
        class_numbers[data['template']].append(data)
        # new_data.append(data)

# print(class_numbers)
final_li = []
for key in class_numbers.keys():
    final_li.extend(sample_thousand_rows(class_numbers[key]))

print(final_li, len(final_li))
random.shuffle(final_li)
# Save the modified dataset
with open('train_b-10classes.json', 'w') as f:
    json.dump(final_li, f)

# def plot_histogram(data_dict):
#     categories = list(data_dict.keys())
#     frequencies = list(data_dict.values())

#     # Plot the histogram
#     plt.bar(categories, frequencies)
    
#     # Add labels and title
#     plt.xlabel('Categories')
#     plt.ylabel('Frequency')
#     plt.title('Histogram')
#     plt.xticks(categories, rotation='vertical')

#     # Show the plot
#     plt.show()

# # Call the function with your data dictionary
# plot_histogram(class_numbers)
