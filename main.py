import sys
from src.mod.DataManager import DataManager
from src.mod.RpsAgent import RpsAgent

def determine_winner(user_move: str, ai_move: str, rules: dict) -> str:
    """Evaluates the winner based on game rules."""
    if user_move == ai_move:
        return "DRAW"
    elif ai_move in rules[user_move]:
        return "AI_WINS"
    else:
        return "USER_WINS"

def main():
    print("------------------------------------------")
    print("  ROCK PAPER SCISSORS - MARKOV v1.0    ")
    print("------------------------------------------")
    
    # LOAD DATA AND INITIALIZE AGENT
    manager = DataManager()
    agent = RpsAgent(manager)
    
    options = {"1": "Rock", "2": "Paper", "3": "Scissors", "4": "Lizard", "5": "Spock"}
    
    print(f"[*] History loaded: {len(manager.samples)} samples")
    print("[*] Ready to play. Press 0 to save and exit.")
    
    while True:
        ai_move = agent.get_action()
        
        print("\nChoose your move:")
        print(" [1] Rock")
        print(" [2] Paper")
        print(" [3] Scissors")
        print(" [4] Lizard")
        print(" [5] Spock")
        print(" [0] Exit")
        
        choice = input(">> ").strip()
        
        if choice == "0":
            print("\n[!] Saving history and exiting...")
            break
            
        if choice not in options:
            print("[?] Invalid option. Try again.")
            continue
            
        user_move = options[choice]
        result = determine_winner(user_move, ai_move, agent.rules)
        
        print("\n------------------------------------------")
        print(f" USER: {user_move}")
        print(f" AI  : {ai_move}")
        print("------------------------------------------")
        
        if result == "DRAW":
            print(" RESULT: [ IT'S A TIE ]")
        elif result == "AI_WINS":
            print(" RESULT: [ AI WINS (X_X) ]")
        else:
            print(" RESULT: [ YOU WIN! (^_^) ]")
        print("------------------------------------------")
        
        # SHARE YOUR MOVE WITH THE MARKOV CHAIN
        agent.update(user_move)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Forced exit. Data might not be saved.")
        sys.exit(0)