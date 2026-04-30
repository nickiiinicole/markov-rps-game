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
        self.rules = {
            "Rock": ["Paper", "Spock"],
            "Paper": ["Scissors", "Lizard"],
            "Scissors": ["Rock", "Spock"],
            "Lizard": ["Rock", "Scissors"],
            "Spock": ["Paper", "Lizard"]
        }
        self.valid_moves = list(self.rules.keys())
        
        # Transformation categorical <-> numerical for processing  
        self.encode = {"Rock": 0, "Paper": 1, "Scissors": 2, "Lizard": 3, "Spock": 4}
        self.decode = {0: "Rock", 1: "Paper", 2: "Scissors", 3: "Lizard", 4: "Spock"}
        
        # Initialize Pomegranate ML Model
        self.model = MarkovChain(k=1)
        # k is the order of the Markov Chain, 1 means it considers only the last move for prediction

        self.is_trained = False
        
        self._train_model()

    def update(self, current_move: str)-> None:
        """Saves the user's move to disk and retrains the AI."""
        self.data.add_sample(current_move)
        self._train_model()

    def _prepare_dataset(self) -> Optional[torch.Tensor]:
        """Extracts history from JSON and transforms it into 3D PyTorch Tensors."""
        history = list(self.data.samples)
        
        if len(history) < 2:
            return None
            
        # Inject a base history with all 5 options [0, 1, 2, 3, 4] at the beginning.
        # This prevents PyTorch out-of-bounds errors.
        encoded_history = [0, 1, 2, 3, 4] + [self.encode[move] for move in history]
        
        # Pomegranate strictly requires 3D tensors: 
        # (batch_size, sequence_length, n_features)
        # We transform our 1D array into a 3D shape (1, length, 1)
        tensor_1d = torch.tensor(encoded_history)
        tensor_3d = tensor_1d.view(1, -1, 1)
        
        return tensor_3d

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
        
        counter_moves: list[str] = self.rules[predicted_move]
        
        return random.choice(counter_moves)