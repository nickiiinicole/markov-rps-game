import json
import os
# Using deque ensures memory safety by limiting size and removing oldest items when full
from collections import deque

class DataManager:
    """
    Handles file persistence and memory limits.
    """
    
    def __init__(self, filename="move_history.json", max_samples=1000):
        self.filename = filename
        self.max_samples = max_samples
        # Memory safety: deque automatically discards old items when is full
        self.samples = deque(maxlen=self.max_samples)
        self.load_data()

    def add_sample(self, move):
        """Appends a new move and triggers a save operation."""
        self.samples.append(move)
        self.save_data()

    def save_data(self):
        """Saves current memory state to a JSON file."""
        try:
            with open(self.filename, 'w') as f:
                json.dump(list(self.samples), f)
        except IOError as e:
            print(f"[Error] Persistence failed: {e}")

    def load_data(self):
        """Loads historical data if the file exists."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.samples = deque(data, maxlen=self.max_samples)
            except (json.JSONDecodeError, IOError):
                print("[Warning] Data file is invalid. Initializing empty history.")