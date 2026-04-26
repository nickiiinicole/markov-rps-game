import os
import pytest
from src.mod.DataManager import DataManager
from src.mod.RpsAgent import RpsAgent

def test_agent_predicts_obvious_pattern():
    
    test_file = "test_agent_history.json"
    manager = DataManager(filename=test_file, max_samples=10)
    
    # Pruebo com un patron
    pattern = ["Rock", "Paper", "Rock", "Paper", "Rock"]
    for move in pattern:
        manager.add_sample(move)
        
    agent = RpsAgent(manager)
    # Como la última jugada fue "Rock", el modelo (si funciona bien) mirará 
    # su matriz y verá que SIEMPRE que hay "Rock", luego viene "Paper".
    # Por tanto, el agente predecirá "Paper", y para ganarle decidirá sacar "Scissors".
    action = agent.get_action()
    
    # ASSERT 
    assert action == "Scissors", f"Expected 'Scissors', but got '{action}'"
    
    if os.path.exists(test_file):
        os.remove(test_file)

def test_agent_random_fallback_without_history():
    test_file = "test_empty_history.json"
    manager = DataManager(filename=test_file, max_samples=10)
    agent = RpsAgent(manager)
    
    action = agent.get_action()
    
    assert action in ["Rock", "Paper", "Scissors"]
    
    if os.path.exists(test_file):
        os.remove(test_file)