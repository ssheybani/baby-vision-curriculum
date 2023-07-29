import json
from collections import defaultdict
import matplotlib.pyplot as plt
# Load the dataset annotations
with open('train_ann_b-10classes.json', 'r') as f:
    dataset = json.load(f)

print(len(dataset))

# Mapping of existing class labels to new class labels
class_numbers = defaultdict(int)

# new_data = []
# Convert class labels
for data in dataset:
        class_numbers[data['template']] += 1
        # new_data.append(data)

print(class_numbers)
# print(new_data, len(new_data))
# Save the modified dataset
# with open('validation-10classes.json', 'w') as f:
#     json.dump(new_data, f)

def plot_histogram(data_dict):
    categories = list(data_dict.keys())
    frequencies = list(data_dict.values())

    # Plot the histogram
    plt.bar(categories, frequencies)
    
    # Add labels and title
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.xticks(categories, rotation='vertical')

    # Show the plot
    plt.show()

# Call the function with your data dictionary
plot_histogram(class_numbers)
