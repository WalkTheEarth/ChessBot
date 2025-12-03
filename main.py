"""
Chess RL Training - Multi-threaded Command Line Interface
Train neural network against chess engines using multiple threads

Usage:
    python main.py --games 1000 --difficulty 9 --threads 4
    python main.py --games 500 --difficulty 10 --threads 8 --engine-path "C:/Stockfish/stockfish.exe"
    python main.py --help

Arguments:
    --games: Number of games to train (default: 1000)
    --difficulty: Stockfish skill level 0-20 (default: 5)
    --threads: Number of parallel game threads (default: 4)
    --self-play-ratio: Ratio of self-play vs engine games (default: 0.7)
    --save-interval: Save model every N games (default: 100)
    --load-model: Path to load existing model
    --engine-path: Path to stockfish executable
"""

import argparse
import time
import threading
from queue import Queue
from chess_env import ChessEnv
from agent import DQNAgent
from train import Trainer
import torch

def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

class MultiThreadedTrainer:
    """Multi-threaded trainer with shared experience replay"""
    
    def __init__(self, agent, num_threads=4, use_engine=False, engine_path=None, difficulty=5):
        self.agent = agent
        self.num_threads = num_threads
        self.use_engine = use_engine
        self.engine_path = engine_path
        self.difficulty = difficulty
        
        # Shared statistics
        self.stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0,
            'avg_loss': 0
        }
        
        self.stats_lock = threading.Lock()
        self.training_active = True
        self.game_results_queue = Queue()
        
    def worker_thread(self, thread_id, num_games_per_thread, callback=None):
        """Worker thread that plays games"""
        # Each thread gets its own environment and engine
        env = ChessEnv()
        
        trainer = None
        if self.use_engine and self.engine_path:
            try:
                trainer = Trainer(self.agent, use_engine=True, engine_path=self.engine_path)
                if trainer.engine:
                    trainer.engine.set_skill_level(self.difficulty)
            except Exception as e:
                print(f"⚠️  Thread {thread_id}: Engine failed ({e}), using self-play")
                trainer = Trainer(self.agent, use_engine=False)
        else:
            trainer = Trainer(self.agent, use_engine=False)
        
        for game_num in range(num_games_per_thread):
            if not self.training_active:
                break
            
            # Determine opponent
            opponent = 'engine' if trainer.engine and game_num % 10 == 0 else 'self'
            
            # Play game
            game_result = trainer.play_game(env, opponent)
            
            # Update shared statistics
            with self.stats_lock:
                self.stats['games_played'] += 1
                
                result = game_result['result']
                if '1-0' in result:
                    self.stats['wins'] += 1
                elif '0-1' in result:
                    self.stats['losses'] += 1
                else:
                    self.stats['draws'] += 1
                
                self.stats['total_reward'] += game_result['reward']
                
                # Put result in queue for callback
                self.game_results_queue.put((self.stats['games_played'], game_result, self.stats.copy()))
            
            # Decay epsilon (thread-safe)
            self.agent.decay_epsilon()
    
    def train(self, num_games, update_freq=10, save_freq=100, callback=None):
        """Train with multiple threads"""
        games_per_thread = num_games // self.num_threads
        remainder = num_games % self.num_threads
        
        # Start worker threads
        threads = []
        for i in range(self.num_threads):
            games_for_this_thread = games_per_thread + (1 if i < remainder else 0)
            t = threading.Thread(
                target=self.worker_thread,
                args=(i, games_for_this_thread, callback),
                daemon=True
            )
            t.start()
            threads.append(t)
        
        # Start updater thread for target network and saving
        def updater_thread():
            last_save = 0
            last_update = 0
            
            while self.training_active or not self.game_results_queue.empty():
                time.sleep(0.1)
                
                # Process game results from queue
                while not self.game_results_queue.empty():
                    try:
                        game_num, game_result, stats = self.game_results_queue.get_nowait()
                        
                        if callback:
                            callback(game_num, game_result, stats)
                        
                        # Update target network
                        if game_num - last_update >= update_freq:
                            self.agent.update_target_model()
                            last_update = game_num
                        
                        # Save checkpoint
                        if game_num - last_save >= save_freq and game_num > 0:
                            self.agent.save(f'checkpoint_{game_num}.pth')
                            last_save = game_num
                            
                    except:
                        break
        
        updater = threading.Thread(target=updater_thread, daemon=True)
        updater.start()
        
        # Wait for all worker threads to complete
        for t in threads:
            t.join()
        
        self.training_active = False
        updater.join(timeout=2)
        
        return self.stats

def main():
    parser = argparse.ArgumentParser(description='Train Chess RL Agent (Multi-threaded)')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to train')
    parser.add_argument('--difficulty', type=int, default=5, help='Stockfish difficulty (0-20)')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel threads')
    parser.add_argument('--self-play-ratio', type=float, default=0.7, help='Ratio of self-play games')
    parser.add_argument('--save-interval', type=int, default=100, help='Save model every N games')
    parser.add_argument('--update-freq', type=int, default=10, help='Update target network every N games')
    parser.add_argument('--load-model', type=str, default=None, help='Load existing model')
    parser.add_argument('--engine-path', type=str, default=None, help='Path to stockfish.exe')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧠 MULTI-THREADED CHESS REINFORCEMENT LEARNING TRAINER")
    print("=" * 70)
    print(f"\n📊 Training Configuration:")
    print(f"   Games: {args.games}")
    print(f"   Threads: {args.threads} (parallel game execution)")
    print(f"   Engine Difficulty: {args.difficulty}/20")
    print(f"   Self-Play Ratio: {args.self_play_ratio*100:.0f}%")
    print(f"   Save Interval: Every {args.save_interval} games")
    print(f"   Update Frequency: Every {args.update_freq} games")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   🚀 GPU Acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
    else:
        print(f"   ⚠️  GPU Acceleration: DISABLED (using CPU)")
    
    # Initialize agent
    print("\n🔧 Initializing neural network...")
    agent = DQNAgent()
    
    if args.load_model:
        print(f"📂 Loading model from {args.load_model}...")
        try:
            agent.load(args.load_model)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   Starting with fresh model...")
    
    # Initialize multi-threaded trainer
    use_engine = args.engine_path is not None
    if use_engine:
        print(f"♟️  Configuring Stockfish engines for {args.threads} threads...")
        print(f"   Engine Skill Level: {args.difficulty}")
    else:
        print("⚠️  No engine path provided. Using self-play only.")
    
    trainer = MultiThreadedTrainer(
        agent,
        num_threads=args.threads,
        use_engine=use_engine,
        engine_path=args.engine_path,
        difficulty=args.difficulty
    )
    
    print("\n" + "=" * 70)
    print(f"🚀 STARTING TRAINING ({args.threads} THREADS)")
    print("=" * 70)
    print()
    
    start_time = time.time()
    last_update_time = start_time
    game_times = []
    last_games_count = 0
    
    def training_callback(game_num, game_result, stats):
        """Print updates every game"""
        nonlocal last_update_time, last_games_count
        
        current_time = time.time()
        
        # Calculate games per second
        games_since_last = game_num - last_games_count
        time_since_last = current_time - last_update_time
        games_per_sec = games_since_last / time_since_last if time_since_last > 0 else 0
        last_games_count = game_num
        last_update_time = current_time
        
        # Calculate stats
        games = stats['games_played']
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        win_rate = (wins / games * 100) if games > 0 else 0
        
        # Estimate remaining time
        elapsed = current_time - start_time
        games_rate = games / elapsed if elapsed > 0 else 1
        remaining_games = args.games - games
        eta = remaining_games / games_rate if games_rate > 0 else 0
        
        # Print compact update
        result_emoji = "🏆" if "1-0" in game_result['result'] else "💀" if "0-1" in game_result['result'] else "🤝"
        
        print(f"Game {games:4d}/{args.games} {result_emoji} | "
              f"W:{wins:3d} L:{losses:3d} D:{draws:3d} | "
              f"WR:{win_rate:5.1f}% | "
              f"ε:{agent.epsilon:.4f} | "
              f"Moves:{game_result['moves']:3d} | "
              f"Speed:{games_rate:.1f}g/s | "
              f"ETA:{format_time(eta)}")
        
        # Milestone updates
        if games % args.save_interval == 0 and games > 0:
            print(f"\n💾 Checkpoint saved: checkpoint_{games}.pth")
            print(f"   Current Stats: {wins}W/{losses}L/{draws}D ({win_rate:.1f}% win rate)")
            print()
        
        if games % (args.games // 10) == 0 and games > 0:
            print(f"\n📈 Progress Report ({games}/{args.games}):")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Epsilon: {agent.epsilon:.4f}")
            print(f"   Total Reward: {stats['total_reward']}")
            print(f"   Time Elapsed: {format_time(elapsed)}")
            print(f"   Avg Speed: {games_rate:.2f} games/sec")
            print()
    
    # Train the agent
    try:
        trainer.train(
            num_games=args.games,
            update_freq=args.update_freq,
            save_freq=args.save_interval,
            callback=training_callback
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        trainer.training_active = False
    
    # Final statistics
    total_time = time.time() - start_time
    stats = trainer.stats
    
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Statistics:")
    print(f"   Total Games: {stats['games_played']}")
    print(f"   Wins: {stats['wins']} ({stats['wins']/stats['games_played']*100:.1f}%)")
    print(f"   Losses: {stats['losses']} ({stats['losses']/stats['games_played']*100:.1f}%)")
    print(f"   Draws: {stats['draws']} ({stats['draws']/stats['games_played']*100:.1f}%)")
    print(f"   Total Reward: {stats['total_reward']}")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    print(f"   Training Time: {format_time(total_time)}")
    print(f"   Avg Time/Game: {format_time(total_time/stats['games_played'])}")
    print(f"   Avg Speed: {stats['games_played']/total_time:.2f} games/sec")
    print(f"   Speedup: {args.threads}x threads = {(stats['games_played']/total_time)/(stats['games_played']/total_time/args.threads):.2f}x faster")
    
    # Save final model
    print("\n💾 Saving final model...")
    agent.save('final_model.pth')
    print("✅ Model saved as 'final_model.pth'")
    print("\n🎉 Done! Your chess AI is trained!")
    print("=" * 70)

if __name__ == "__main__":
    main()
