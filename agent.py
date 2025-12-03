import random
from collections import deque
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np
from model import ChessNet
from chess_env import ChessEnv

class DQNAgent:
    """Deep Q-Network agent for chess"""
    
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.9995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = ChessNet().to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Thread safety locks
        self.train_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.epsilon_lock = threading.Lock()
        
    def select_move(self, env: ChessEnv, fast_mode=True) -> chess.Move:
        """Select move using epsilon-greedy policy
        
        Args:
            fast_mode: If True, use batch evaluation for speed
        """
        legal_moves = env.get_legal_moves()
        
        if not legal_moves:
            return None
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Exploitation - BATCH evaluate all legal moves (MUCH faster)
        if fast_mode and len(legal_moves) > 1:
            next_states = []
            for move in legal_moves:
                temp_board = env.board.copy()
                temp_board.push(move)
                temp_env = ChessEnv()
                temp_env.board = temp_board
                next_states.append(temp_env.get_state())
            
            # Batch evaluation
            states_tensor = torch.from_numpy(np.array(next_states)).float().to(self.device)
            with torch.no_grad():
                values = self.model(states_tensor).squeeze().cpu().numpy()
            
            if len(legal_moves) == 1:
                return legal_moves[0]
            
            best_idx = np.argmax(values)
            return legal_moves[best_idx]
        else:
            # Original slow method (fallback)
            best_move = None
            best_value = float('-inf')
            
            for move in legal_moves:
                temp_board = env.board.copy()
                temp_board.push(move)
                temp_env = ChessEnv()
                temp_env.board = temp_board
                next_state = temp_env.get_state()
                
                state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    value = self.model(state_tensor).item()
                
                if value > best_value:
                    best_value = value
                    best_move = move
            
            return best_move
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory (thread-safe)"""
        with self.memory_lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from memory (thread-safe)"""
        with self.memory_lock:
            if len(self.memory) < self.batch_size:
                return 0
            batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first (more efficient)
        states_np = np.array([s[0] for s in batch])
        rewards_np = np.array([s[2] for s in batch])
        next_states_np = np.array([s[3] for s in batch])
        dones_np = np.array([s[4] for s in batch])
        
        # Convert to tensors
        states = torch.from_numpy(states_np).float().to(self.device)
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        dones = torch.from_numpy(dones_np).float().to(self.device)
        
        # Training with lock
        with self.train_lock:
            # Current Q values
            current_q = self.model(states).squeeze()
            
            # Target Q values
            with torch.no_grad():
                next_q = self.target_model(next_states).squeeze()
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            # Compute loss and update
            loss = self.criterion(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
    
    def update_target_model(self):
        """Update target network with current model weights (thread-safe)"""
        with self.train_lock:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate (thread-safe)"""
        with self.epsilon_lock:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']