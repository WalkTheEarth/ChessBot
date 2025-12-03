# ChessBot API Documentation

This document provides detailed API documentation for all classes and methods in the ChessBot project.

## Table of Contents

1. [Core AI Components](#core-ai-components)
   - [DQNAgent](#dqagent)
   - [ChessNet](#chessnet)
   - [ChessEnv](#chessenv)
2. [Training Components](#training-components)
   - [Trainer](#trainer)
   - [UCIEngine](#uciengine)
3. [User Interface](#user-interface)
   - [ChessTUI](#chesstui)
4. [Main Entry Points](#main-entry-points)
   - [MultiThreadedTrainer](#multithreadedtrainer)

---

## Core AI Components

### DQNAgent

The main Deep Q-Network agent that learns to play chess through reinforcement learning.

#### Constructor
```python
DQNAgent(learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01)
```

**Parameters:**
- `learning_rate` (float): Learning rate for Adam optimizer (default: 0.001)
- `gamma` (float): Discount factor for future rewards (default: 0.99)
- `epsilon` (float): Initial exploration rate (default: 1.0)
- `epsilon_decay` (float): Rate of epsilon decay (default: 0.9995)
- `epsilon_min` (float): Minimum exploration rate (default: 0.01)

#### Methods

##### `select_move(env: ChessEnv, fast_mode: bool = True) -> chess.Move`
Selects the best move using epsilon-greedy policy.

**Parameters:**
- `env` (ChessEnv): Current chess environment
- `fast_mode` (bool): Use batch evaluation for speed (default: True)

**Returns:**
- `chess.Move`: Selected move or None if no legal moves

**Behavior:**
- With probability `epsilon`: returns random legal move (exploration)
- With probability `1-epsilon`: returns best predicted move (exploitation)
- Uses batch processing when `fast_mode=True` for better performance

##### `remember(state, action, reward, next_state, done)`
Stores experience in replay memory buffer.

**Parameters:**
- `state`: Current board state (12×8×8 tensor)
- `action`: Move taken (chess.Move)
- `reward`: Reward received (float)
- `next_state`: Resulting board state (12×8×8 tensor)
- `done`: Game termination flag (bool)

##### `replay() -> float`
Trains the neural network on a batch of experiences.

**Returns:**
- `float`: Training loss

**Behavior:**
- Samples random batch from memory buffer
- Computes Q-values using current and target networks
- Updates network weights using Adam optimizer
- Thread-safe implementation

##### `update_target_model()`
Updates target network with current model weights.

**Behavior:**
- Copies weights from main network to target network
- Called periodically to stabilize training
- Thread-safe implementation

##### `decay_epsilon()`
Decreases exploration rate over time.

**Behavior:**
- Multiplies epsilon by decay factor
- Ensures epsilon doesn't go below minimum value
- Thread-safe implementation

##### `save(path: str)`
Saves model checkpoint to file.

**Parameters:**
- `path` (str): File path for saving model

**Saves:**
- Model state dictionary
- Optimizer state dictionary
- Current epsilon value

##### `load(path: str)`
Loads model checkpoint from file.

**Parameters:**
- `path` (str): File path to load model from

**Loads:**
- Model state dictionary
- Optimizer state dictionary
- Epsilon value

---

### ChessNet

Convolutional Neural Network for chess position evaluation.

#### Constructor
```python
ChessNet()
```

#### Architecture
```
Input: (batch_size, 12, 8, 8)
├── Conv2d(12 → 64, kernel=3×3, padding=1) + ReLU
├── Conv2d(64 → 128, kernel=3×3, padding=1) + ReLU
├── Conv2d(128 → 128, kernel=3×3, padding=1) + ReLU
├── Flatten: (batch_size, 8192)
├── Linear(8192 → 512) + ReLU
├── Dropout(0.3)
├── Linear(512 → 256) + ReLU
└── Linear(256 → 1)
Output: (batch_size, 1) - Q-value
```

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, 12, 8, 8)

**Returns:**
- `torch.Tensor`: Q-value tensor of shape (batch_size, 1)

---

### ChessEnv

Chess environment wrapper for reinforcement learning.

#### Constructor
```python
ChessEnv()
```

#### Methods

##### `reset() -> np.ndarray`
Resets the board to starting position.

**Returns:**
- `np.ndarray`: Initial board state (12×8×8 tensor)

##### `get_state() -> np.ndarray`
Gets current board state as neural network input.

**Returns:**
- `np.ndarray`: Board state tensor (12×8×8)

**State Representation:**
- Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- Each channel: 8×8 binary matrix (1 = piece present, 0 = empty)

##### `make_move(move: chess.Move) -> Tuple[np.ndarray, float, bool]`
Makes a move and returns the result.

**Parameters:**
- `move` (chess.Move): Move to make

**Returns:**
- `Tuple[np.ndarray, float, bool]`: (next_state, reward, done)

**Reward System:**
- Win: +1000
- Loss: -1000
- Draw: 0
- Illegal move: -100
- Move limit exceeded: -500

##### `get_legal_moves() -> List[chess.Move]`
Gets list of legal moves.

**Returns:**
- `List[chess.Move]`: List of legal moves

##### `is_game_over() -> bool`
Checks if game is over.

**Returns:**
- `bool`: True if game is over

##### `get_result() -> str`
Gets game result.

**Returns:**
- `str`: Game result ("1-0", "0-1", "1/2-1/2")

---

## Training Components

### Trainer

Training loop for chess RL agent.

#### Constructor
```python
Trainer(agent: DQNAgent, use_engine=False, engine_path=None)
```

**Parameters:**
- `agent` (DQNAgent): DQN agent to train
- `use_engine` (bool): Whether to use chess engine (default: False)
- `engine_path` (str): Path to chess engine executable (default: None)

#### Methods

##### `play_game(env: ChessEnv, opponent='self', max_moves=200) -> dict`
Plays one complete game.

**Parameters:**
- `env` (ChessEnv): Chess environment
- `opponent` (str): 'self' for self-play, 'engine' for engine opponent
- `max_moves` (int): Maximum moves before declaring draw

**Returns:**
- `dict`: Game result with keys:
  - `result`: Game outcome string
  - `moves`: Number of moves played
  - `reward`: Total reward earned
  - `avg_loss`: Average training loss

##### `train(num_games=1000, update_freq=10, save_freq=100, callback=None)`
Trains the agent for specified number of games.

**Parameters:**
- `num_games` (int): Number of games to play
- `update_freq` (int): Update target network every N games
- `save_freq` (int): Save checkpoint every N games
- `callback` (callable): Function called after each game

**Returns:**
- `dict`: Training statistics

---

### UCIEngine

Interface for UCI chess engines (Stockfish/Sailfish).

#### Constructor
```python
UCIEngine(engine_path="C:\\stockfish\\stockfish.exe", skill_level=1)
```

**Parameters:**
- `engine_path` (str): Path to engine binary
- `skill_level` (int): Engine strength (0-20, lower is weaker)

#### Methods

##### `set_skill_level(level: int)`
Sets engine difficulty.

**Parameters:**
- `level` (int): Skill level (0-20)

##### `get_best_move(board: chess.Board) -> Optional[chess.Move]`
Gets best move from engine.

**Parameters:**
- `board` (chess.Board): Current board position

**Returns:**
- `Optional[chess.Move]`: Best move or None if no move available

##### `close()`
Closes engine connection.

---

## User Interface

### ChessTUI

Terminal UI for chess RL training (experimental).

#### Constructor
```python
ChessTUI()
```

#### Methods

##### `render_board(board_str: str) -> Panel`
Renders chess board as Rich panel.

##### `render_stats(stats: dict) -> Table`
Renders training statistics as Rich table.

##### `run()`
Main TUI loop.

---

## Main Entry Points

### MultiThreadedTrainer

Multi-threaded trainer with shared experience replay.

#### Constructor
```python
MultiThreadedTrainer(agent, num_threads=4, use_engine=False, engine_path=None, difficulty=5)
```

**Parameters:**
- `agent` (DQNAgent): DQN agent to train
- `num_threads` (int): Number of parallel threads
- `use_engine` (bool): Whether to use chess engine
- `engine_path` (str): Path to chess engine
- `difficulty` (int): Engine difficulty level

#### Methods

##### `train(num_games, update_freq=10, save_freq=100, callback=None)`
Trains with multiple threads.

**Parameters:**
- `num_games` (int): Total number of games
- `update_freq` (int): Target network update frequency
- `save_freq` (int): Checkpoint save frequency
- `callback` (callable): Progress callback function

**Returns:**
- `dict`: Final training statistics

---

## Usage Examples

### Basic Training
```python
from agent import DQNAgent
from train import Trainer
from chess_env import ChessEnv

# Initialize components
agent = DQNAgent()
env = ChessEnv()
trainer = Trainer(agent)

# Train for 1000 games
stats = trainer.train(num_games=1000)
print(f"Final win rate: {stats['wins']/stats['games_played']*100:.1f}%")
```

### Engine Opposition Training
```python
from agent import DQNAgent
from train import Trainer

# Initialize with engine
agent = DQNAgent()
trainer = Trainer(agent, use_engine=True, engine_path="C:/Stockfish/stockfish.exe")

# Train against engine
stats = trainer.train(num_games=1000)
```

### Multi-threaded Training
```python
from main import MultiThreadedTrainer
from agent import DQNAgent

# Initialize multi-threaded trainer
agent = DQNAgent()
trainer = MultiThreadedTrainer(agent, num_threads=6, use_engine=True, 
                              engine_path="C:/Stockfish/stockfish.exe", difficulty=8)

# Train with progress callback
def progress_callback(game_num, result, stats):
    print(f"Game {game_num}: {result['result']}")

trainer.train(num_games=2000, callback=progress_callback)
```

### Model Loading and Saving
```python
from agent import DQNAgent

# Create and train agent
agent = DQNAgent()
# ... training code ...

# Save model
agent.save('my_chess_model.pth')

# Load model
new_agent = DQNAgent()
new_agent.load('my_chess_model.pth')
```

---

## Thread Safety

The ChessBot implementation is designed for multi-threaded training with the following thread-safe components:

- **DQNAgent**: All methods are thread-safe with appropriate locking
- **Experience Replay**: Shared memory buffer with thread-safe access
- **Model Updates**: Synchronized training updates
- **Epsilon Decay**: Thread-safe exploration rate management

## Performance Considerations

- **Batch Processing**: Use `fast_mode=True` in `select_move()` for better performance
- **Memory Management**: Monitor replay buffer size (default: 10,000 experiences)
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Thread Count**: Optimal performance with 4-8 threads depending on CPU cores
