import chess
from typing import Optional
try:
    from stockfish import Stockfish
except ImportError:
    Stockfish = None

class UCIEngine:
    """Interface for UCI chess engines (Stockfish/Sailfish)"""
    
    def __init__(self, engine_path="C:\stockfish\stockfish.exe", skill_level=1):
        """
        Initialize UCI engine
        
        Args:
            engine_path: Path to engine binary
            skill_level: Engine strength (0-20, lower is weaker)
        """
        self.engine = Stockfish(path=engine_path, parameters={
            "Skill Level": skill_level
        })
        self.skill_level = skill_level
        
    def set_skill_level(self, level: int):
        """Set engine difficulty (0-20)"""
        self.skill_level = level
        self.engine.set_skill_level(level)
        
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move from engine"""
        self.engine.set_fen_position(board.fen())
        move_str = self.engine.get_best_move()
        
        if move_str:
            return chess.Move.from_uci(move_str)
        return None
    
    def close(self):
        """Close engine"""
        pass  # Stockfish wrapper handles this