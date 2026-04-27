# Markov Chain - Rock Paper Scissors AI

A terminal-based implementation of the Rock Paper Scissors game where the opponent is a Markov Chain AI. The system learns from user behavior patterns to predict and counter future moves.

## Technical Overview
The project utilizes the Pomegranate library for Markov Chain modeling and PyTorch for tensor operations. It implements a first-order Markov Chain that calculates transition probabilities based on the user's historical data.

## Features
- Dynamic learning: The AI retrains itself after every round.
- Persistence: Match history is stored in 'move_history.json' to retain knowledge across sessions.
- Error Handling: Implements Laplace-style smoothing to manage new states and ensures 3D tensor compatibility for the ML model.

## Installation
Dependencies are managed via 'uv'. To install the required packages, run:

1. Create a virtual environment:
```bash
uv venv
```

2. Install the required dependencies:
```bash
    uv pip install -r requirements.txt
```

3. Running the Application
```bash
    -----linux/Mac Os-----
    source .venv/bin/activate
    uv run python main.py
    -----Windows-----
    .venv\Scripts\activate
    python main.py
```