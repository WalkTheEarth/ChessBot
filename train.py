import time
from chess_env import ChessEnv
from agent import DQNAgent
from uci_handler import UCIEngine

class Trainer:
    """Training loop for chess RL agent"""
    
    def __init__(self, agent: DQNAgent, use_engine=False, engine_path=None):
        self.agent = agent
        self.use_engine = use_engine
        self.engine = None
        
        if use_engine and engine_path:
            try:
                self.engine = UCIEngine(engine_path, skill_level=1)
            except:
                self.engine = None
        
        self.stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0,
            'avg_loss': 0
        }
        
    def play_game(self, env: ChessEnv, opponent='self', max_moves=200) -> dict:
        """
        Play one game
        
        Args:
            opponent: 'self' for self-play, 'engine' for engine opponent
            max_moves: Maximum moves before declaring draw (prevents infinite games)
        """
        env.reset()
        total_reward = 0
        moves = 0
        losses = []
        
        while not env.is_game_over() and moves < max_moves:
            state = env.get_state()
            
            # Agent's turn
            move = self.agent.select_move(env)
            if move is None:
                break
                
            next_state, reward, done = env.make_move(move)
            total_reward += reward
            moves += 1
            
            self.agent.remember(state, move, reward, next_state, done)
            
            if done:
                break
            
            # Opponent's turn
            if opponent == 'engine' and self.engine:
                opp_move = self.engine.get_best_move(env.board)
                if opp_move:
                    _, opp_reward, done = env.make_move(opp_move)
                    total_reward -= opp_reward  # Opponent's reward is negative for us
                    moves += 1
            elif opponent == 'self':
                # Self-play: agent plays both sides
                opp_move = self.agent.select_move(env)
                if opp_move:
                    env.make_move(opp_move)
                    moves += 1
            
            # Train on batch every few moves (not every move)
            if moves % 5 == 0 and len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.replay()
                losses.append(loss)
        
        # Determine result and apply punishment for exceeding move limit
        if moves >= max_moves:
            result = "1/2-1/2"  # Draw by move limit
            total_reward -= 500  # Punishment for taking too long
        else:
            result = env.get_result()
            
        game_result = {
            'result': result,
            'moves': moves,
            'reward': total_reward,
            'avg_loss': sum(losses) / len(losses) if losses else 0
        }
        
        # Update stats
        self.stats['games_played'] += 1
        if '1-0' in result:
            self.stats['wins'] += 1
        elif '0-1' in result:
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
        self.stats['total_reward'] += total_reward
        
        return game_result
    
    def train(self, num_games=1000, update_freq=10, save_freq=100, 
              callback=None):
        """
        Train the agent
        
        Args:
            num_games: Number of games to play
            update_freq: Update target network every N games
            save_freq: Save model every N games
            callback: Function called after each game with stats
        """
        env = ChessEnv()
        
        for game in range(num_games):
            # Switch between self-play and engine play
            opponent = 'engine' if self.engine and game % 10 == 0 else 'self'
            
            game_result = self.play_game(env, opponent)
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Update target network
            if game % update_freq == 0:
                self.agent.update_target_model()
            
            # Save checkpoint
            if game % save_freq == 0 and game > 0:
                self.agent.save(f'checkpoint_{game}.pth')
            
            # Callback for UI updates
            if callback:
                callback(game, game_result, self.stats)
        
        return self.stats