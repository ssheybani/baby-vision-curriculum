import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
# Load the dataset annotations
dataset = pd.read_csv('test-answers-10classes.csv', header=None, delimiter=',')
print(dataset.head())
print(dataset)

# Convert class labels
class_numbers = defaultdict(int)
for i in range(len(dataset)):
    class_numbers[dataset.iloc[i,1]] += 1


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
    plt.savefig("histogram_test.png")
    # Show the plot
    plt.show()

# Call the function with your data dictionary
plot_histogram(class_numbers)
