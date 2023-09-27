import pandas as pd
from collections import defaultdict
# Load the dataset annotations
dataset = pd.read_csv('test-answers.csv', header=None, delimiter=';')
# print(dataset.head())
# print(dataset)

# Mapping of existing class labels to new class labels
class_mapping = {
    "Something falling like a rock": "Dropping something",
    "Something falling like a feather or paper" : "Dropping something",
    "Throwing something": "Dropping something",
    "Throwing something onto a surface": "Dropping something",
    "Throwing something in the air and letting it fall": "Dropping something",
    "Pushing something from right to left" : "Moving something from right to left",
    "Pulling something from right to left" : "Moving something from right to left",
    "Pulling something from left to right" : "Moving something from left to right",
    "Pushing something from left to right" : "Moving something from left to right",
    "Picking something up" : "Picking something up",
    "Lifting something up completely without letting it drop down" : "Picking something up",
    "Moving something up" : "Picking something up",
    "Lifting something with something on it" : "Picking something up",
    "Taking something from somewhere" : "Picking something up",
    "Taking one of many similar things on the table" : "Picking something up",
    "Taking something out of something" : "Picking something up",
    "Putting something next to something" : "Putting something",
    "Putting something onto something" : "Putting something",
    "Putting something on a surface" : "Putting something",
    "Putting something similar to other things that are already on the table Putting something behind something" : "Putting something",
    "Putting something, something and something on the table" : "Putting something",
    "Putting something and something on the table" : "Putting something",
    "Putting something on a flat surface without letting it roll" : "Putting something",
    "Putting something that can’t roll onto a slanted surface, so it stays where it is" : "Putting something",
    "Pouring something into something" : "Pouring something",
    "Pouring something onto something" : "Pouring something",
    "Pouring something out of something" : "Pouring something",
    "Pouring something into something until it overflows" : "Pouring something",
    "Trying to pour something into something, but missing so it spills next to it" : "Pouring something",
    "Poking something so that it falls over" : "Poking something",
    "Poking something so lightly that it doesn’t or almost doesn’t move" : "Poking something",
    "Poking a stack of something so the stack collapses" : "Poking something",
    "Poking a stack of something without the stack collapsing" : "Poking something",
    "Tearing something into two pieces" : "Tearing something", 
    "Tearing something just a little bit" : "Tearing something",
    "Holding something" : "Holding something",
    "Holding something in front of something" : "Holding something",
    "Showing something on top of something" : "Showing something (almost no hand)",
    "Showing something behind something" : "Showing something (almost no hand)",
    "Showing something next to something" : "Showing something] (almost no hand)"
}
new_data = defaultdict(list)
# Convert class labels
for i in range(len(dataset)):
    if dataset.iloc[i,1] in class_mapping:
        dataset.iloc[i,1] = class_mapping[dataset.iloc[i,1]]
        new_data["id"].append(dataset.iloc[i,0])
        new_data["template"].append(dataset.iloc[i,1])
new_data = pd.DataFrame(new_data)
print(new_data.head(), len(new_data))
# Save the modified dataset
new_data.to_csv('test-answers-10classes.csv', index=False, header=False)