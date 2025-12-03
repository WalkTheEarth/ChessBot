# 🧠 ChessBot - AI Chess Engine Trainer

A sophisticated reinforcement learning system that trains neural networks to play chess using Deep Q-Learning, self-play, and engine opposition.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start training (self-play)
python main.py --games 100 --threads 2

# Train against Stockfish
python main.py --games 1000 --difficulty 5 --engine-path "C:/Stockfish/stockfish.exe" --threads 4
```

## 📚 Documentation

### **Getting Started**
- **[Quick Start Guide](docs/QUICK_START.md)** - Get training in under 5 minutes
- **[Full Documentation](docs/README.md)** - Comprehensive guide with all features
- **[API Reference](docs/API_DOCUMENTATION.md)** - Complete API documentation

### **Key Features**
- 🧠 **Deep Q-Network (DQN)** with CNN architecture
- ⚡ **Multi-threaded training** (up to 8x speedup)
- 🎮 **Self-play and engine opposition** training modes
- 📊 **Real-time progress monitoring** with Rich UI
- 💾 **Automatic checkpointing** and model saving

## 🎯 Training Examples

### Basic Self-Play
```bash
python main.py --games 1000 --threads 4
```

### Engine Opposition
```bash
python main.py --games 1000 --difficulty 8 --engine-path "C:/Stockfish/stockfish.exe" --threads 6
```

### Mixed Training
```bash
python main.py --games 2000 --difficulty 10 --self-play-ratio 0.7 --engine-path "C:/Stockfish/stockfish.exe" --threads 8
```

## 📁 Project Structure

```
ChessBot/
├── 🧠 Core AI Components
│   ├── agent.py          # DQN agent with experience replay
│   ├── model.py          # CNN neural network architecture
│   └── chess_env.py      # Chess environment & state representation
│
├── 🎮 Training & Execution
│   ├── main.py           # Multi-threaded CLI training interface
│   ├── train.py          # Training loop and game simulation
│   └── uci_handler.py    # Stockfish/UCI engine integration
│
├── 🎯 Play Against AI
│   ├── play_chess.py     # GUI chess game (tkinter)
│   ├── play_chess_cli.py # Command-line chess game
│   ├── launch_game.py    # Game launcher with dependency check
│   └── play_chess.bat    # Windows batch file launcher
│
├── 📚 Documentation
│   ├── README.md         # Comprehensive documentation
│   ├── QUICK_START.md    # Quick start guide
│   └── API_DOCUMENTATION.md # Complete API reference
│
├── 💾 Model Storage
│   ├── checkpoint_*.pth  # Training checkpoints
│   ├── final_model.pth   # Final trained model
│   └── Model1/, Model2/, Model3/  # Alternative model versions
│
└── requirements.txt      # Python dependencies
```

## 🎮 Usage

### **Play Against Your Trained AI**

#### **Easy Launch (Recommended)**
```bash
# Windows: Double-click play_chess.bat
# Or run the launcher
python launch_game.py
```

#### **Direct Launch**
```bash
# GUI Version (Recommended)
pip install Pillow cairosvg  # Install GUI dependencies
python play_chess.py

# Command Line Version
python play_chess_cli.py
```

**Features:**
- 🎯 **Model Selection**: Choose from your trained checkpoints
- 🎚️ **AI Difficulty**: Adjust AI strength (0=best, 0.5=random)
- ↩️ **Undo Moves**: Take back moves if needed
- 📊 **Move History**: See all moves played
- 🎨 **Visual Board**: Click to move pieces (GUI version)

### **Training Your AI**

#### **Command Line Arguments**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--games` | int | 1000 | Total number of training games |
| `--threads` | int | 4 | Parallel game threads (1-8 recommended) |
| `--difficulty` | int | 5 | Stockfish skill level (0-20, higher = stronger) |
| `--self-play-ratio` | float | 0.7 | Ratio of self-play vs engine games (0.0-1.0) |
| `--load-model` | str | None | Path to existing model checkpoint |
| `--engine-path` | str | None | Path to Stockfish executable |

### Training Progress

During training, you'll see real-time updates:
```
Game  250/1000 🏆 | W: 85 L: 120 D: 45 | WR: 34.0% | ε:0.7750 | Moves: 47 | Speed:3.2g/s | ETA:3.9m
```

**Legend:**
- 🏆/💀/🤝 = Win/Loss/Draw
- WR = Win Rate
- ε = Exploration rate (starts at 1.0, decays to 0.01)
- Speed = Games per second

## 🧠 Neural Network Architecture

- **Input**: 12×8×8 tensor (6 piece types × 2 colors)
- **Architecture**: 3-layer CNN + fully connected layers
- **Output**: Single Q-value for position evaluation
- **Training**: Experience replay with target network

## 📈 Expected Performance

| Training Phase | Games | Win Rate vs Difficulty 5 | Key Learning |
|----------------|-------|---------------------------|--------------|
| **Exploration** | 0-500 | ~20-30% | Basic rules, piece movement |
| **Development** | 500-2000 | ~30-45% | Tactical patterns, piece coordination |
| **Refinement** | 2000-5000 | ~45-60% | Strategic planning, endgames |
| **Mastery** | 5000+ | ~60%+ | Advanced tactics, opening theory |

## 🐛 Troubleshooting

### Common Issues
- **"ModuleNotFoundError"**: Run `pip install -r requirements.txt`
- **"Engine connection failed"**: Check Stockfish path or use self-play
- **"CUDA out of memory"**: Reduce threads or use CPU-only mode
- **Training not progressing**: Increase games or adjust difficulty

### Getting Help
1. **Read the [Quick Start Guide](docs/QUICK_START.md)** for immediate help
2. **Check the [Full Documentation](docs/README.md)** for comprehensive troubleshooting
3. **Review the [API Documentation](docs/API_DOCUMENTATION.md)** for technical details

## 🚀 Future Enhancements

- [ ] Policy gradient methods (PPO/A3C)
- [ ] Monte Carlo Tree Search integration
- [ ] Opening book integration
- [ ] Web interface for visualization
- [ ] ELO rating tracking
- [ ] Multi-GPU support

## 📄 License

This project is released under the **MIT License**. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **PyTorch** - Deep learning framework
- **python-chess** - Chess library and UCI protocol
- **Stockfish** - Open source chess engine
- **Rich** - Beautiful terminal interfaces
- **AlphaZero** - Inspiration for self-play RL

---

## 🎉 Ready to Train Your Chess AI?

```bash
# Quick start command
python main.py --games 100 --threads 4

# Full training session
python main.py --games 2000 --threads 6 --difficulty 8 --engine-path "C:/Stockfish/stockfish.exe"
```

**Happy Training! May your neural network achieve chess mastery! ♟️🧠🚀**

*For detailed documentation, see the [docs/](docs/) folder.*
