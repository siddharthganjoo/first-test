import json
import streamlit as st
# Initial preset thresholds
thresholds = {
   "Blood": 80,
   "Yolk": 120,
   "Feces": 130,
   "Bruised": 20,
   "OpenCrack": 20,
   "Crack": 175,
   "Wrinkle": 6500,
   "Feather": 100,
   "CalciumSpot": 100,
   "Dust": 150,
   "Group_Dirty": 20,
   "Group_Damaged": 20,
   "Group_ShellDeviation": 1000
}
# Streamlit UI
st.title("False Negative Learning Agent")
st.write("Upload a False Negative JSONL file, and the agent will suggest new thresholds.")
# File uploader
uploaded_file = st.file_uploader("Upload FN JSONL File", type="jsonl")
if uploaded_file is not None:
   try:
       # Read the FN file
       data = []
       with uploaded_file as file:
           for line in file:
               egg = json.loads(line)
               data.append(egg)
       if not data:
           st.error("The uploaded file is empty or not in the expected format.")
       else:
           st.success(f"Loaded {len(data)} FN eggs.")
           # Step 1: Analyze the FNs and Identify Labels Causing Most Issues
           label_adjustments = {key: 0 for key in thresholds.keys()}
           for egg in data:
               for label, value in egg.items():
                   if label in thresholds and value > thresholds[label]:
                       label_adjustments[label] += 1  # Count how many times each label is above threshold
           # Step 2: Adjust Thresholds Based on Frequency of FNs
           adjusted_thresholds = thresholds.copy()
           for label, count in label_adjustments.items():
               if count > 0:  # If a label caused FNs, increase the threshold slightly
                   increase_factor = 1 + (count / len(data)) * 0.05  # Increase threshold based on FN frequency
                   adjusted_thresholds[label] = round(thresholds[label] * increase_factor, 2)
           # Step 3: Display Adjusted Thresholds
           st.subheader("Adjusted Thresholds Based on FN Data")
           st.json(adjusted_thresholds)
           st.write("The model has suggested new threshold values to reduce False Negatives.")
   except Exception as e:
       st.error(f"An error occurred: {e}")