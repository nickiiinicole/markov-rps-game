import random

# Standard Rules: Key beats Value
GAME_RULES = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper"
}

VALID_MOVES = list(GAME_RULES.keys())

class RPSAgent:
    """
    Model-Based Agent: It remembers the opponent's moves to predict the next one.
    """
    def __init__(self):
        # Internal State: History of opponent's moves
        self.history = []

    def update(self, move):
        """Sensor: Updates the internal state with the new observation."""
        if move in VALID_MOVES:
            self.history.append(move)

    def get_action(self):
        """
        Decision Rule: 
        1. Analyze history to find the Mode (most frequent move).
        2. Select the counter-move that beats it.
        """
        # If no history is available, play random
        if not self.history:
            return random.choice(VALID_MOVES)

        # Find the opponent's most frequent move (The Mode)
        most_frequent = max(set(self.history), key=self.history.count)
        
        # Determine the move that beats the most frequent one
        # We look for the key in GAME_RULES that beats 'most_frequent'
        counter_move = None
        for winner, loser in GAME_RULES.items():
            if loser == most_frequent:
                counter_move = winner
                break
        
        return counter_move if counter_move else random.choice(VALID_MOVES)

def determine_winner(user_move, computer_move):
    """Pure function to determine the winner."""
    if user_move == computer_move:
        return "Tie"
    
    # Check if user's move beats computer's move
    if GAME_RULES[user_move] == computer_move:
        return "User"
    else:
        return "Computer"

def main():
    print("--- Classic Rock, Paper, Scissors AI ---")
    print(f"Options: {', '.join(VALID_MOVES)}")
    print("Type 'Exit' to quit.\n")

    agent = RPSAgent()
    
    # Control variable to avoid 'while True'
    keep_playing = True

    while keep_playing:
        user_input = input("Your move: ").capitalize()

        if user_input == "Exit":
            print("Game over. Goodbye!")
            keep_playing = False
        
        elif user_input in VALID_MOVES:
            # Agent makes a decision
            computer_move = agent.get_action()
            
            print(f"AI chooses: {computer_move}")
            
            # Determine result
            result = determine_winner(user_input, computer_move)
            print(f"Win Result :D : {result}")
            print("-" * 30)

            # Update Agent's memory (Learning step)
            agent.update(user_input)
            
        else:
            print("Invalid move. Please try again.")

if __name__ == "__main__":
    main()