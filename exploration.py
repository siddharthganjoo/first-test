import streamlit as st
import json
# Preset Thresholds with Original Labels
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
   "Group_Schaalafwijking": 1000
}
# Adjustment Strategies: Increase by 25%, 50%, 75%, and 100% of MaxDeviation
adjustment_strategies = [0.25, 0.5, 0.75, 1.0]
# Streamlit App Layout
st.title("Threshold Adjustment Explorer for 1% FNs")
st.write("Explore how much to adjust thresholds to accept 1% False Negatives (FNs).")
st.write("Adjustments are made by a percentage of MaxDeviation for each label.")
# File Uploader
uploaded_file = st.file_uploader("Upload 1% FN JSONL File", type="jsonl")
# Process the file if uploaded
if uploaded_file is not None:
   data = []
   # Read the JSONL File
   try:
       for line in uploaded_file:
           egg = json.loads(line)
           data.append(egg)
       st.success(f"Loaded {len(data)} eggs from the uploaded file.")
   except Exception as e:
       st.error(f"Error reading JSONL file: {e}")
       st.stop()
   # Intelligent Exploration of Adjustments
   st.subheader("Intelligent Exploration of Threshold Adjustments")
   adjusted_thresholds = thresholds.copy()
   for label, preset_threshold in thresholds.items():
       st.markdown(f"### Adjustments for `{label}`")
       st.write(f"Preset Threshold: `{preset_threshold}`")
       # Check the 1% FNs for this label
       fn_values = [egg.get(label, 0) for egg in data]
       fn_values = [value for value in fn_values if value > 0]  # Only consider non-zero values
       if not fn_values:
           st.info(f"No FNs for `{label}`.")
           continue
       st.write(f"1% FNs for `{label}`: {fn_values}")
       # Explore different adjustment strategies
       strategy_results = []
       for strategy in adjustment_strategies:
           # Calculate Adjustment using MaxDeviation
           adjusted_values = []
           for egg in data:
               max_deviation = egg.get("MaxDeviation", 0)
               adjusted_threshold = round(preset_threshold + (max_deviation * strategy), 2)
               adjusted_values.append(adjusted_threshold)
           # Check how many FNs would be accepted with this adjustment
           accepted_count = sum(1 for i, value in enumerate(fn_values) if value <= adjusted_values[i])
           total_count = len(fn_values)
           acceptance_rate = round((accepted_count / total_count) * 100, 2)
           # Collect the results for display
           strategy_results.append({
               "Strategy": f"{int(strategy * 100)}% of MaxDeviation",
               "Adjusted Thresholds": adjusted_values,
               "FNs Accepted": accepted_count,
               "Total FNs": total_count,
               "Acceptance Rate": acceptance_rate
           })
       # Display Exploration Results
       st.write(f"**Exploration Results for `{label}`:**")
       for result in strategy_results:
           st.write(f"- Strategy: `{result['Strategy']}`")
           st.write(f"  - Adjusted Thresholds: `{result['Adjusted Thresholds']}`")
           st.write(f"  - FNs Accepted: `{result['FNs Accepted']} / {result['Total FNs']} ({result['Acceptance Rate']}%)`")
       # Find the Best Adjustment Strategy
       best_strategy = max(strategy_results, key=lambda x: x['Acceptance Rate'])
       adjusted_thresholds[label] = max(best_strategy["Adjusted Thresholds"])
   # Display Adjusted Thresholds
   st.subheader("Adjusted Thresholds to Minimize 1% FNs")
   st.json(adjusted_thresholds)