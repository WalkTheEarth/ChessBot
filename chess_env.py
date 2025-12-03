import chess
import numpy as np
from typing import Optional, Tuple

class ChessEnv:
    """Chess environment for RL training"""
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        
    def reset(self):
        """Reset the board to starting position"""
        self.board = chess.Board()
        self.move_history = []
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Convert board to neural network input (8x8x12 tensor)"""
        state = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_type = piece_map[piece.piece_type]
                offset = 0 if piece.color == chess.WHITE else 6
                state[piece_type + offset, rank, file] = 1
                
        return state
    
    def get_legal_moves(self):
        """Get list of legal moves"""
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> Tuple[np.ndarray, float, bool]:
        """
        Make a move and return (next_state, reward, done)
        Reward is 0 for ongoing game, will be set at end
        """
        if move not in self.board.legal_moves:
            # Illegal move penalty
            return self.get_state(), -100, True
        
        self.board.push(move)
        self.move_history.append(move)
        
        done = self.board.is_game_over()
        reward = 0
        
        if done:
            result = self.board.result()
            if result == "1-0":  # White wins
                reward = 1000 if self.board.turn == chess.BLACK else -1000
            elif result == "0-1":  # Black wins
                reward = 1000 if self.board.turn == chess.WHITE else -1000
            else:  # Draw
                reward = 0
                
        return self.get_state(), reward, done
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.board.is_game_over()
    
    def get_result(self) -> str:
        """Get game result"""
        return self.board.result()
    
    def __str__(self):
        """String representation of board"""
        return str(self.board)


# ====================
# model.py
# ====================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    """Convolutional Neural Network for chess position evaluation"""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Q-value output
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, 12, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x