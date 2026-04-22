import os
import json
import pytest
from src.mod.DataManager import DataManager
from src.mod.RpsAgent import RpsAgent



def create_sample(filename, moves):
    with open(filename, 'w') as f:
        json.dump(moves, f)

def test_read_json():
    test_filename = "mock_history_rock.json"
    mock_moves = ["Rock", "Rock", "Rock", "Rock", "Rock"]
    create_sample(test_filename, mock_moves)
    
    try:
        manager = DataManager(filename=test_filename)
        agent = RpsAgent(manager)
        
        # La IA verá que la última fue 'Rock' y la probabilidad de que siga 'Rock' es alta.
        ai_move = agent.get_action()
        
        # Para ganar a 'Rock', debe sacar 'Paper'
        assert ai_move == "Paper", f"La IA debería haber sacado Paper para ganar al Rock obsesivo, pero sacó {ai_move}"
        
    finally:
        if os.path.exists(test_filename):
            os.remove(test_filename)

def test_reads_json2():
    test_filename = "mock_history_cycle.json"
    # Papel -> Tijera -> Papel -> Tijera...
    mock_moves = ["Paper", "Scissors", "Paper", "Scissors", "Paper", "Scissors"]
    create_sample(test_filename, mock_moves)
    
    try:
        manager = DataManager(filename=test_filename)
        agent = RpsAgent(manager)
        
        ai_move = agent.get_action()
        
        # Última fue Scissors, predice Paper, la IA saca Scissors para ganar.
        assert ai_move == "Scissors", f"La IA debería haber sacado Scissors para ganar al ciclo, pero sacó {ai_move}"
        
    finally:
        if os.path.exists(test_filename):
            os.remove(test_filename)


