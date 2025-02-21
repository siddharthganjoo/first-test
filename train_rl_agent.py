import streamlit as st
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from fn_rl_env import FNThresholdEnv  # Import our custom Gym environment
# Initial thresholds
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
st.title("RL-Based False Negative Threshold Optimization")
st.write("Upload FN JSONL file and let the RL model learn the best threshold values.")
# File uploader
uploaded_file = st.file_uploader("Upload FN JSONL File", type="jsonl")
if uploaded_file is not None:
   try:
       # Save the uploaded file
       fn_file = "fn_data.jsonl"
       with open(fn_file, "wb") as f:
           f.write(uploaded_file.getbuffer())
       st.success("FN file uploaded successfully!")
       # ✅ Create Gym environment
       env = FNThresholdEnv(fn_file, thresholds)
       # ✅ Train RL agent
       model = PPO("MlpPolicy", env, verbose=1)
       model.learn(total_timesteps=5000)
       # ✅ Get optimized thresholds
       obs, _ = env.reset()
       for _ in range(10):  # Let the model adjust 10 times
           action, _ = model.predict(obs)
           # ✅ FIX: Handle the 5-return values correctly
           obs, reward, terminated, truncated, info = env.step(action)
           # ✅ If the episode is terminated, reset the environment
           if terminated or truncated:
               obs, _ = env.reset()
       # ✅ Extract new thresholds
       new_thresholds = {key: obs[i] for i, key in enumerate(thresholds.keys())}
       # ✅ Display new thresholds
       st.subheader("Optimized Thresholds from RL Agent")
       st.json(new_thresholds)
   except Exception as e:
       st.error(f"Error: {e}")