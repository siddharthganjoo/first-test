import gymnasium as gym
import numpy as np
import jsonlines
class FNThresholdEnv(gym.Env):
   def __init__(self, fn_file, thresholds):
       super(FNThresholdEnv, self).__init__()
       # Load FN data
       self.fn_data = []
       with jsonlines.open(fn_file, "r") as reader:
           for line in reader:
               self.fn_data.append(line)
       # Store thresholds
       self.thresholds = thresholds.copy()
       self.current_thresholds = thresholds.copy()
       # ✅ Define observation space correctly (One value per threshold + FN count)
       self.observation_space = gym.spaces.Box(
           low=0, high=10000, shape=(len(thresholds) + 1,), dtype=np.float32  # +1 for FN count
       )
       # ✅ Actions: Only increase thresholds, no decrease
       self.action_space = gym.spaces.Discrete(len(thresholds))  # One action per label (increase only)
       # ✅ Track previous FN count to calculate reward correctly
       self.previous_fn_count = self._calculate_fn_count()
   def step(self, action):
       label_list = list(self.current_thresholds.keys())
       label_idx = action  # ✅ No more division by 2, since we only increase
       # ✅ Only increase the threshold (no decreasing allowed)
       self.current_thresholds[label_list[label_idx]] += 5  # Increase by 5
       # ✅ Calculate new FN count
       new_fn_count = self._calculate_fn_count()
       # ✅ Define Reward: Reward is positive if FN decreases, negative if FN increases
       reward = self.previous_fn_count - new_fn_count  # Lower FN = Positive Reward
       # ✅ Update previous FN count
       self.previous_fn_count = new_fn_count
       # ✅ Fix observation shape: Only include thresholds + FN count
       observation = np.array(list(self.current_thresholds.values()) + [new_fn_count], dtype=np.float32)
       return observation, reward, False, False, {}
   def reset(self, seed=None, options=None):
       # ✅ Reset to stored `self.thresholds`
       self.current_thresholds = self.thresholds.copy()
       # ✅ Initial FN count
       initial_fn_count = self._calculate_fn_count()
       self.previous_fn_count = initial_fn_count  # ✅ Reset previous FN count
       # ✅ Fix observation shape: Match expected size
       observation = np.array(list(self.current_thresholds.values()) + [initial_fn_count], dtype=np.float32)
       return observation, {}
   def _calculate_fn_count(self):
       """Calculate the FN count based on current thresholds."""
       fn_count = 0
       for egg in self.fn_data:
           for label, value in egg.items():
               if label in self.current_thresholds:
                   threshold = self.current_thresholds[label]
                   # ✅ Ignore values that are much lower than threshold
                   if value < threshold:
                       continue
                   if value > threshold:
                       fn_count += 1
                       break  # Only count once per egg
       return fn_count