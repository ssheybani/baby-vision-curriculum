import json
import csv
from sklearn.preprocessing import LabelEncoder

# Replace 'path_to_json_file.json' with the actual path to your JSON file
input_json_file = '/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_labels/train_ann_b-10classes.json'
output_csv_file = '/N/project/baby_vision_curriculum/benchmarks/ssv2/easy_labels/train_easy10.csv'
# 
# Read JSON data from the file
with open(input_json_file, 'r') as f:
    data = json.load(f)

# Prepare the list of dictionaries for CSV writing
rows = [{"fname": entry["id"] + ".webm", "label": entry["template"]} for entry in data]

# Map the 'template' column to integers
# label_encoder = LabelEncoder()
# labels = [entry["template"] for entry in data]
# label_encoder.fit(labels)
# for entry in rows:
#     entry["label"] = label_encoder.transform([entry["label"]])[0]

# Write CSV data to the file
with open(output_csv_file, 'w', newline='') as f:
    fieldnames = ['fname', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print(f"Modified CSV file '{output_csv_file}' has been created successfully.")
