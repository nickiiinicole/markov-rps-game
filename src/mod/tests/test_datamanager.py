import os
import pytest
from src.mod.DataManager import DataManager

def test_data_manager():
    test_file = "test_data.json"
    manager = DataManager(filename=test_file, max_samples=3)
    
    manager.add_sample("Rock")
    manager.add_sample("Paper")
    
    assert len(manager.samples) == 2
    assert os.path.exists(test_file)
    
    if os.path.exists(test_file):
        os.remove(test_file)