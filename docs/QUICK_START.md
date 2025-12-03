# ChessBot Quick Start Guide

Get your chess AI training in under 5 minutes!

## 🚀 30-Second Setup

### 1. Install Dependencies
```bash
pip install torch numpy python-chess rich stockfish
```

### 2. Download Stockfish (Optional)
**Windows:** Download from [stockfishchess.org](https://stockfishchess.org/download/)
**Linux:** `sudo apt install stockfish`
**macOS:** `brew install stockfish`

### 3. Start Training
```bash
python main.py --games 100 --threads 2
```

That's it! Your AI is now learning to play chess! 🎉

---

## 🎯 First Training Session

### Test Your Setup (2 minutes)
```bash
# Quick test with minimal resources
python main.py --games 10 --threads 1
```

**Expected Output:**
```
🧠 MULTI-THREADED CHESS REINFORCEMENT LEARNING TRAINER
======================================================================
📊 Training Configuration:
   Games: 10
   Threads: 1 (parallel game execution)
   Engine Difficulty: 5/20
   Self-Play Ratio: 70%
   Save Interval: Every 100 games
   Update Frequency: Every 10 games
   🚀 GPU Acceleration: ENABLED (NVIDIA GeForce RTX 3080)

🔧 Initializing neural network...
♟️  Configuring Stockfish engines for 1 threads...
   Engine Skill Level: 5

======================================================================
🚀 STARTING TRAINING (1 THREADS)
======================================================================

Game    1/10 🏆 | W:  1 L:  0 D:  0 | WR:100.0% | ε:0.9995 | Moves: 45 | Speed:0.8g/s | ETA:11.2s
Game    2/10 💀 | W:  1 L:  1 D:  0 | WR: 50.0% | ε:0.9990 | Moves: 38 | Speed:1.2g/s | ETA:6.7s
...
```

### Your First Real Training (10 minutes)
```bash
# Balanced training session
python main.py --games 500 --threads 4 --difficulty 3
```

**What to Expect:**
- **Games 0-100**: Random play, learning basic rules
- **Games 100-300**: Developing basic tactics
- **Games 300-500**: Improving strategic play

---

## 🎮 Training Scenarios

### Beginner: Self-Play Only
```bash
python main.py --games 1000 --threads 4
```
- **Best for**: Learning the basics
- **Time**: ~30 minutes
- **Result**: Basic chess understanding

### Intermediate: Engine Opposition
```bash
python main.py --games 1000 --difficulty 5 --engine-path "C:/Stockfish/stockfish.exe" --threads 6
```
- **Best for**: Tactical improvement
- **Time**: ~20 minutes
- **Result**: Competitive play

### Advanced: Mixed Training
```bash
python main.py --games 2000 --difficulty 8 --self-play-ratio 0.7 --engine-path "C:/Stockfish/stockfish.exe" --threads 8
```
- **Best for**: Strategic mastery
- **Time**: ~45 minutes
- **Result**: Strong chess AI

---

## 📊 Understanding the Output

### Progress Display
```
Game  250/1000 🏆 | W: 85 L: 120 D: 45 | WR: 34.0% | ε:0.7750 | Moves: 47 | Speed:3.2g/s | ETA:3.9m
```

**Legend:**
- **🏆/💀/🤝**: Win/Loss/Draw
- **W/L/D**: Win/Loss/Draw counts
- **WR**: Win rate percentage
- **ε**: Exploration rate (starts at 1.0, decreases to 0.01)
- **Moves**: Average moves per game
- **Speed**: Games per second
- **ETA**: Estimated time remaining

### Milestone Updates
```
💾 Checkpoint saved: checkpoint_100.pth
   Current Stats: 85W/120L/45D (34.0% win rate)

📈 Progress Report (250/1000):
   Win Rate: 34.0%
   Epsilon: 0.7750
   Total Reward: 12500
   Time Elapsed: 2.1m
   Avg Speed: 3.15 games/sec
```

---

## 🎯 Success Indicators

### Good Training Progress
- ✅ **Win rate increasing** over time (20% → 40% → 60%+)
- ✅ **Epsilon decreasing** (1.0 → 0.01)
- ✅ **Games completing** without errors
- ✅ **Checkpoints saving** regularly

### Warning Signs
- ⚠️ **Win rate stuck** at very low levels (<10%)
- ⚠️ **Games ending** in illegal moves frequently
- ⚠️ **Memory errors** or crashes
- ⚠️ **No progress** after 500+ games

---

## 🔧 Quick Fixes

### Training Too Slow?
```bash
# Increase threads (but not more than CPU cores)
python main.py --games 1000 --threads 6
```

### Out of Memory?
```bash
# Reduce threads and batch size
python main.py --games 1000 --threads 2
```

### Engine Not Working?
```bash
# Use self-play only
python main.py --games 1000 --threads 4
```

### Want to Resume Training?
```bash
# Load from checkpoint
python main.py --games 1000 --load-model checkpoint_500.pth --threads 4
```

---

## 🎉 Next Steps

### After Your First Training
1. **Check your model**: `ls -la *.pth` (should see checkpoint files)
2. **Analyze results**: Look at final win rate and training time
3. **Experiment**: Try different difficulty levels and thread counts
4. **Scale up**: Train for longer with more games

### Advanced Usage
- **Custom hyperparameters**: Edit `agent.py` for learning rates, etc.
- **Different architectures**: Modify `model.py` for network changes
- **Engine tuning**: Adjust Stockfish parameters in `uci_handler.py`
- **Visualization**: Use the experimental TUI in `tui.py`

---

## 🆘 Need Help?

### Common Issues
1. **"ModuleNotFoundError"**: Run `pip install -r requirements.txt`
2. **"Engine connection failed"**: Check Stockfish path or use self-play
3. **"CUDA out of memory"**: Reduce threads or use CPU-only mode
4. **Training not progressing**: Increase games or adjust difficulty

### Getting Support
- **Read the full README.md** for comprehensive documentation
- **Check API_DOCUMENTATION.md** for detailed API reference
- **Review troubleshooting section** in README.md
- **Create an issue** with your system details and error messages

---

## 🏆 Training Goals

### Beginner Goals (First 1000 games)
- [ ] AI learns basic piece movement
- [ ] Win rate reaches 20-30%
- [ ] No illegal moves
- [ ] Games complete successfully

### Intermediate Goals (1000-5000 games)
- [ ] Win rate reaches 40-50%
- [ ] Develops basic tactics
- [ ] Shows strategic thinking
- [ ] Beats lower difficulty engines

### Advanced Goals (5000+ games)
- [ ] Win rate reaches 60%+
- [ ] Master-level play
- [ ] Complex strategic planning
- [ ] Competitive against strong engines

---

**Happy Training! Your chess AI journey starts now! ♟️🧠🚀**
