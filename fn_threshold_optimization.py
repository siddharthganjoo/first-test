import streamlit as st
import jsonlines
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from fn_rl_env import FNThresholdEnv  

# ✅ Initial Thresholds
thresholds = {
    "Bloed": 80.0,
    "Eigeel": 120.0,
    "Mest": 130.0,
    "Kneus": 20.0,
    "Openbreuk": 20.0,
    "Scheur": 175.0,
}

# ✅ Define Max Thresholds (+20% Ceiling)
max_thresholds = {key: round(value * 1.2, 2) for key, value in thresholds.items()}

# ✅ Store Threshold History for Visualization
threshold_history = {key: [value] for key, value in thresholds.items()}

# ✅ Function to Extract FNs
def extract_fns(file_path, thresholds, percentage=1.0):
    fns = []
    with jsonlines.open(file_path, "r") as reader:
        for egg_list in reader:
            for egg in egg_list:
                label = egg.get("Label")
                value = egg.get("Value", 0)
                if label in thresholds:
                    threshold = thresholds[label]
                    deviation = ((value - threshold) / threshold) * 100
                    if 0 < deviation <= percentage:
                        fns.append(egg_list)
                        break  
    return fns

# ✅ --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Egg Sorting Optimization - Vencomatic",
    page_icon="🥚",
    layout="wide"
)

# ✅ Sidebar with branding
st.sidebar.image("venco.png", use_container_width=True)
st.sidebar.title("Egg Sorting Optimization")
st.sidebar.info("Upload daily batch data to optimize egg sorting thresholds.")

# ✅ Main Section Title
st.title("🐔 Meggsius Select Automatic Calibration")
st.markdown("""
**Welcome to the Vencomatic Group Meggsius Select System.**  
This system learns from False Negatives (FNs) to adjust sorting thresholds per batch.
""")

# ✅ FN Assumption & Learning Explanation
st.markdown("## 📝 How FN Detection & Learning Works")
st.info("""
🔍 **False Negatives (FNs) Assumption**
- If an egg is **just above the threshold (+1%)**, we **assume** it is a False Negative.
- These eggs are manually **reinserted by the farmer** to help the machine learn.
  
🤖 **How the Machine Learns**
- The machine **analyzes these FNs** and **adjusts thresholds slightly** to accept them in the future.
- **Over multiple days, thresholds adapt** based on real farm conditions.
- **Goal:** Minimize False Negatives while keeping sorting accuracy high.
""")

# ✅ File Upload Section
uploaded_files = st.file_uploader("Upload 16 JSONL Files", type="jsonl", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 16:
    file_paths = []
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = f"day_{i+1}.jsonl"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    current_thresholds = thresholds.copy()
    fn_counts = []

    for day in range(1, 17):
        st.subheader(f"📅 Processing Day {day}...")

        fns_day = extract_fns(file_paths[day-1], current_thresholds, percentage=1.0)
        fn_counts.append(len(fns_day))

        # ✅ Store Preset Thresholds Before RL Adjustment
        old_thresholds = current_thresholds.copy()

        if day == 1:
            # ✅ Day 1: Increase Thresholds by 1% (Initial Step)
            for key in current_thresholds.keys():
                current_thresholds[key] = round(current_thresholds[key] * 1.01, 2)  
        else:
            best_thresholds = None
            best_fn_count = float('inf')

            for _ in range(10):
                fn_file = f"fn_training_day{day}.jsonl"
                with jsonlines.open(fn_file, "w") as writer:
                    for egg_list in fns_day:
                        writer.write(egg_list)

                env = FNThresholdEnv(fn_file, current_thresholds, max_thresholds)  
                model = PPO("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=5000)

                obs, _ = env.reset()
                for _ in range(10):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, _ = env.reset()

                candidate_thresholds = {key: float(obs[i]) for i, key in enumerate(current_thresholds.keys())}
                candidate_fn_count = len(extract_fns(file_paths[day-1], candidate_thresholds, percentage=1.0))

                if candidate_fn_count < best_fn_count:
                    best_fn_count = candidate_fn_count
                    best_thresholds = candidate_thresholds

            # ✅ Limit Max Threshold Change per Day (+1.0 max)
            for key in best_thresholds.keys():
                best_thresholds[key] = round(min(best_thresholds[key], current_thresholds[key] + 1.0), 2)

            current_thresholds = best_thresholds.copy()
            st.write(f"New RL-Optimized Thresholds for Day {day}: {best_thresholds}")

        # ✅ Store Thresholds for Graphs
        for key in current_thresholds.keys():
            threshold_history[key].append(current_thresholds[key])

    # ✅ 📈 Graph: Threshold Adjustments Over Time
    st.subheader("📊 Threshold Adjustments Over Time")
    for label, values in threshold_history.items():
        plt.plot(range(1, len(values) + 1), values, marker='o', label=label)

    plt.xlabel("Day")
    plt.ylabel("Threshold Value")
    plt.title("Daily Threshold Adjustments")
    plt.legend()
    st.pyplot(plt)

    # ✅ 📊 Final FN Reduction Over Time
    st.subheader("📉 Final FN Reduction Over 16 Days")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(fn_counts) + 1), fn_counts, marker='o', linestyle='-', color="red")
    plt.xlabel("Day")
    plt.ylabel("FN Count")
    plt.title("False Negatives Reduction Over 16 Days")
    st.pyplot(plt)

    # ✅ 📊 Final FN & Threshold Comparison (Day 1 vs Day 16)
    st.subheader("📊 Day 1 vs Day 16 Comparison")

    # ✅ FN Count Comparison
    col1, col2 = st.columns(2)
    col1.metric(label="FN Count on Day 1", value=fn_counts[0])
    col2.metric(label="FN Count on Day 16", value=fn_counts[-1])

    # ✅ Threshold Comparison Table
    final_thresholds = {key: current_thresholds[key] for key in thresholds.keys()}
    df_comparison = {
        "Initial Thresholds": thresholds,
        "Final Thresholds": final_thresholds
    }
    st.table(df_comparison)

    st.success("✅ Threshold optimization completed! Check results above.")
