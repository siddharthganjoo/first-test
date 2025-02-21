import streamlit as st
import jsonlines
import pandas as pd
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from fn_rl_env import FNThresholdEnv  # Import our RL environment
# ✅ Step 1: Define Initial Thresholds (Only Changing These)
thresholds = {
   "Bloed": 80,
   "Eigeel": 120,
   "Mest": 130,
   "Kneus": 20,
   "Openbreuk": 20,
   "Scheur": 175,
}
# ✅ Function to Extract Negatives and FNs
def extract_negatives_and_fns(file_path, thresholds, percentage=1.0):
   negatives = []
   fns = []
   with jsonlines.open(file_path, "r") as reader:
       for egg_list in reader:  # ✅ Each line in JSONL is a LIST of dictionaries
           is_negative = False
           is_fn = False
           max_deviation = 0
           # ✅ Iterate over each dictionary inside the list
           for egg in egg_list:  
               if not isinstance(egg, dict):  # ✅ Ensure it's a dictionary
                   continue  
               label = egg.get("Label")  
               value = egg.get("Value", 0)
               if label in thresholds:
                   threshold = thresholds[label]
                   deviation = ((value - threshold) / threshold) * 100
                   if deviation > 0:  
                       is_negative = True
                       if deviation <= percentage:
                           is_fn = True
                       max_deviation = max(max_deviation, deviation)
           if is_negative:
               negatives.append(egg_list)
           if is_fn:
               fns.append(egg_list)
   return negatives, fns
# ✅ Streamlit UI
st.title("FN Threshold Optimization for 4-Day Egg Batch")
st.write("Upload JSONL files for **Day 1, Day 2, Day 3, and Day 4** to optimize False Negative (FN) thresholds.")
# ✅ Upload Files for 4 Days
uploaded_files = st.file_uploader("Upload 4 JSONL Files (One for Each Day)", type="jsonl", accept_multiple_files=True)
if uploaded_files and len(uploaded_files) == 4:
   try:
       # ✅ Save Files
       file_paths = []
       for i, uploaded_file in enumerate(uploaded_files):
           file_path = f"day_{i+1}.jsonl"
           with open(file_path, "wb") as f:
               f.write(uploaded_file.getbuffer())
           file_paths.append(file_path)
       # ✅ Process Day 1
       st.subheader("Processing Day 1...")
       negatives_day1, fns_day1 = extract_negatives_and_fns(file_paths[0], thresholds, percentage=1.0)
       st.write(f"Day 1 - **Total Negatives**: {len(negatives_day1)}, **1% FN Count**: {len(fns_day1)}")
       # ✅ Increment Thresholds Linearly (e.g., Blood 80 → 81)
       new_thresholds = {key: value + 1 for key, value in thresholds.items()}
       st.write("New Thresholds for Day 2:", new_thresholds)
       # ✅ Process Day 2
       st.subheader("Processing Day 2...")
       negatives_day2, fns_day2 = extract_negatives_and_fns(file_paths[1], new_thresholds, percentage=1.0)
       st.write(f"Day 2 - **Total Negatives**: {len(negatives_day2)}, **1% FN Count**: {len(fns_day2)}")
       # ✅ Train RL Agent Based on FN Changes from Day 2 → Day 3
       st.subheader("Training RL Model (Day 3)...")
       best_thresholds = None
       best_fn_count = float('inf')
       for _ in range(10):  
           fn_file = "fn_training_data.jsonl"
           with jsonlines.open(fn_file, "w") as writer:
               for egg_list in fns_day2:
                   for egg in egg_list:
                       writer.write(egg)
           # ✅ Create RL Environment and Train
           env = FNThresholdEnv(fn_file, new_thresholds)
           model = PPO("MlpPolicy", env, verbose=1)
           model.learn(total_timesteps=5000)
           # ✅ Get RL-Optimized Thresholds
           obs, _ = env.reset()
           for _ in range(10):  
               action, _ = model.predict(obs)
               obs, reward, terminated, truncated, info = env.step(action)
               if terminated or truncated:
                   obs, _ = env.reset()
           candidate_thresholds = {key: float(obs[i]) for i, key in enumerate(new_thresholds.keys())}
           # ✅ Apply these thresholds on Day 3 and check FN count
           negatives_day3, fns_day3 = extract_negatives_and_fns(file_paths[2], candidate_thresholds, percentage=1.0)
           if len(fns_day3) < best_fn_count:
               best_fn_count = len(fns_day3)
               best_thresholds = candidate_thresholds
       for key in best_thresholds.keys():
           best_thresholds[key] = round(min(best_thresholds[key], new_thresholds[key] + 2), 2)
       st.write("RL-Optimized Thresholds for Day 3 (Best Among 10 Runs):", best_thresholds)
       # ✅ Process Day 3 (Apply RL-Optimized Thresholds)
       st.subheader("Processing Day 3...")
       negatives_day3, fns_day3 = extract_negatives_and_fns(file_paths[2], best_thresholds, percentage=1.0)
       st.write(f"Day 3 - **Total Negatives**: {len(negatives_day3)}, **1% FN Count**: {len(fns_day3)}")
       # ✅ Train RL Again for Day 4
       st.subheader("Training RL Model (Day 4)...")
       best_thresholds_day4 = None
       best_fn_count_day4 = float('inf')
       for _ in range(10):  
           fn_file = "fn_training_data_day4.jsonl"
           with jsonlines.open(fn_file, "w") as writer:
               for egg_list in fns_day3:
                   for egg in egg_list:
                       writer.write(egg)
           env = FNThresholdEnv(fn_file, best_thresholds)
           model = PPO("MlpPolicy", env, verbose=1)
           model.learn(total_timesteps=5000)
           obs, _ = env.reset()
           for _ in range(10):  
               action, _ = model.predict(obs)
               obs, reward, terminated, truncated, info = env.step(action)
               if terminated or truncated:
                   obs, _ = env.reset()
           candidate_thresholds_day4 = {key: float(obs[i]) for i, key in enumerate(best_thresholds.keys())}
           negatives_day4, fns_day4 = extract_negatives_and_fns(file_paths[3], candidate_thresholds_day4, percentage=1.0)
           if len(fns_day4) < best_fn_count_day4:
               best_fn_count_day4 = len(fns_day4)
               best_thresholds_day4 = candidate_thresholds_day4
       for key in best_thresholds_day4.keys():
           best_thresholds_day4[key] = round(min(best_thresholds_day4[key], best_thresholds[key] + 2), 2)
       st.write("RL-Optimized Thresholds for Day 4 (Best Among 10 Runs):", best_thresholds_day4)
       st.subheader("Processing Day 4...")
       negatives_day4, fns_day4 = extract_negatives_and_fns(file_paths[3], best_thresholds_day4, percentage=1.0)
       st.write(f"Day 4 - **Total Negatives**: {len(negatives_day4)}, **1% FN Count**: {len(fns_day4)}")
   except Exception as e:
       st.error(f"Error: {e}")