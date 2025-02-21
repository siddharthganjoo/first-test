import json
# Translation mapping for outlier region labels
translation_map = {
   "Bloed": "Blood",
   "Eigeel": "Yolk",
   "Mest": "Feces",
   "Kneus": "Bruised",
   "Openbreuk": "OpenCrack",
   "Scheur": "Crack",
   "Rimpel": "Wrinkle"
   # "Veer": "Feather",
   # "Kalkspot": "CalciumSpot",
   # "Stof": "Dust",
   # "groep_Vervuild": "Group_Dirty",
   # "groep_Beschadigd": "Group_Damaged",
   # "groep_Schaalafwijking": "Group_ShellDeviation"
}
# Thresholds for each label
thresholds = {
   "Blood": 80,
   "Yolk": 120,
   "Feces": 130,
   "Bruised": 20,
   "OpenCrack": 20,
   "Crack": 175,
   "Wrinkle": 6501,
   "Feather": 199090,
   "CalciumSpot": 999999,
   "Dust": 999999,
   "Group_Dirty": 999999,
   "Group_Damaged": 999999,
   "Group_ShellDeviation": 99999
}
# Function to calculate deviation percentage
def calculate_deviation(value, threshold):
   return ((value - threshold) / threshold) * 100
# Initialize a list to hold the 1% FN training batch
fn_1_percent_batch = []
# File paths
input_file = '1_output.jsonl'
output_file = '1st_step_new_threshold.jsonl'
# Read and process the JSONL file line by line
with open(input_file, 'r') as file:
   for line in file:
       # Parse the line as a list of dictionaries
       egg = json.loads(line)
       max_deviation = 0
       is_1_percent_fn = False
       # Translate labels and check against thresholds
       translated_egg = {}
       for item in egg:
           label = item['Label']
           value = item['Value']
           translated_label = translation_map.get(label, label)
           # Check if the translated label has a threshold
           if translated_label in thresholds:
               threshold = thresholds[translated_label]
               deviation = calculate_deviation(value, threshold)
               # Check if deviation is within 1%
               if 0 < deviation <= 1:
                   is_1_percent_fn = True
                   if deviation > max_deviation:
                       max_deviation = deviation
               # Store the translated value
               translated_egg[translated_label] = value
       # If the egg is a 1% FN, add it to the batch
       if is_1_percent_fn:
           translated_egg['MaxDeviation'] = max_deviation
           fn_1_percent_batch.append(translated_egg)
# Save the 1% FN batch to a new JSONL file
with open(output_file, 'w') as outfile:
   for egg in fn_1_percent_batch:
       json.dump(egg, outfile)
       outfile.write('\n')
print(f"1% FN Training batch created with {len(fn_1_percent_batch)} eggs.")
print(f"File saved as {output_file}")