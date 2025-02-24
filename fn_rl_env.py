import gymnasium as gym
import numpy as np
import jsonlines

class FNThresholdEnv(gym.Env):
    def __init__(self, fn_file, thresholds, max_thresholds):
        super(FNThresholdEnv, self).__init__()

        # ✅ Load FN data
        self.fn_data = []
        with jsonlines.open(fn_file, "r") as reader:
            for line in reader:
                self.fn_data.append(line)

        # ✅ Store initial thresholds & allow decimal values
        self.thresholds = {key: float(value) for key, value in thresholds.items()}
        self.current_thresholds = self.thresholds.copy()

        # ✅ Set Maximum Ceiling (Passed as Argument)
        self.threshold_ceiling = max_thresholds.copy()

        # ✅ Observation space: Threshold values + FN count
        self.observation_space = gym.spaces.Box(
            low=0, high=10000, shape=(len(thresholds) + 1,), dtype=np.float32  
        )

        # ✅ Action space: Allow decimal adjustments (-1 to +2)
        self.action_space = gym.spaces.Box(
            low=-1, high=2, shape=(len(thresholds),), dtype=np.float32  
        )

        # ✅ Track previous FN count
        self.previous_fn_count = self._calculate_fn_count()

    def step(self, action):
        label_list = list(self.current_thresholds.keys())

        # ✅ Apply controlled threshold updates (+2 max per step, decimals allowed)
        for i, change in enumerate(action):
            new_value = self.current_thresholds[label_list[i]] + min(max(-1, change), 2)  
            self.current_thresholds[label_list[i]] = min(new_value, self.threshold_ceiling[label_list[i]])  

        # ✅ Calculate new FN count
        new_fn_count = self._calculate_fn_count()

        # ✅ FN reduction percentage
        if self.previous_fn_count > 0:
            fn_reduction_percentage = ((self.previous_fn_count - new_fn_count) / self.previous_fn_count) * 100
        else:
            fn_reduction_percentage = 0

        # ✅ Reward = FN reduction percentage
        reward = fn_reduction_percentage  

        # ✅ Strong penalty if FN count increases
        if new_fn_count > self.previous_fn_count:
            reward -= 5  

        # ✅ Additional Penalty if hitting the Ceiling
        for key, value in self.current_thresholds.items():
            if value >= self.threshold_ceiling[key]:
                reward -= 2  

        # ✅ Store new FN count
        self.previous_fn_count = new_fn_count

        # ✅ Observation: Threshold values + FN count
        observation = np.array(list(self.current_thresholds.values()) + [new_fn_count], dtype=np.float32)

        return observation, reward, False, False, {}

    def reset(self, seed=None, options=None):
        self.current_thresholds = self.thresholds.copy()
        initial_fn_count = self._calculate_fn_count()
        self.previous_fn_count = initial_fn_count  

        observation = np.array(list(self.current_thresholds.values()) + [initial_fn_count], dtype=np.float32)
        return observation, {}

    def _calculate_fn_count(self):
        fn_count = 0

        for egg_list in self.fn_data:  # ✅ Each `egg_list` is a list of dictionaries
            for egg in egg_list:  # ✅ Now loop over individual dictionaries
                if isinstance(egg, dict):  # ✅ Ensure it's a dictionary
                    label = egg.get("Label")
                    value = egg.get("Value", 0)

                    if label in self.current_thresholds:
                        threshold = self.current_thresholds[label]
                        if value > threshold:
                            fn_count += 1  # ✅ Count only once per egg
                            break  # ✅ Stop checking further labels for this egg (to avoid double-counting)

        return fn_count
