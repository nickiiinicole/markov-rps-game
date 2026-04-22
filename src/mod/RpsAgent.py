import random
import torch
from typing import Optional
from pomegranate.markov_chain import MarkovChain

class RpsAgent:
    """
    AI Agent that uses a Markov Chain with Pomegranate
    to predict the user's next move based on their history.
    """
    
    def __init__(self, data_manager)-> None:
         
        self.data = data_manager
        
        # Game Rules
        self.rules = {"Rock": "Paper", "Paper": "Scissors", "Scissors": "Rock"}
        self.valid_moves = list(self.rules.keys())
        
        # Transformation categorical <-> numerical for processing  
        self.encode = {"Rock": 0, "Paper": 1, "Scissors": 2}
        self.decode = {0: "Rock", 1: "Paper", 2: "Scissors"}
        
        # Initialize Pomegranate ML Model
        self.model = MarkovChain(k=1)
        self.is_trained = False
        
        self._train_model()

    def update(self, current_move: str)-> None:
        """Saves the user's move to disk and retrains the AI."""
        self.data.add_sample(current_move)
        self._train_model()

    def _prepare_dataset(self)-> Optional[torch.Tensor]:
        """Extracts history from JSON and transforms it into PyTorch Tensors."""
        history = list(self.data.samples)
        
        if len(history) < 2:
            return None
            
        encoded_history = [self.encode[move] for move in history]
        chain_samples = [[sample] for sample in encoded_history]
        tensor_pairs = [list(pair) for pair in zip(chain_samples, chain_samples[1:])]
        
        return torch.tensor(tensor_pairs)

    def _train_model(self)-> None:
        """Fits the Pomegranate Markov Chain using the tensor dataset."""
        X_tensor = self._prepare_dataset()
        
        if X_tensor is not None:
            # Pomegranate calculates the transition matrix automatically
            self.model.fit(X_tensor)
            self.is_trained = True

    def get_action(self)-> str:
        """Predicts user's next move and returns the winning counter-move."""
        if not self.is_trained or len(self.data.samples) == 0:
            return random.choice(self.valid_moves)

        last_move_str = self.data.samples[-1]
        last_move_idx = self.encode[last_move_str]

        # Extract the transition matrix calculated by Pomegranate
        transition_matrix = self.model.distributions[1].probs[0]
        
        # Find the index with the highest probability
        predicted_idx = int(torch.argmax(transition_matrix[last_move_idx]).item())
        predicted_move = self.decode[predicted_idx]
        
        return self.rules[predicted_move]