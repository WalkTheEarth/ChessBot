"""
ChessBot - Command Line Chess Game
Play against your trained AI in the terminal
"""

import chess
from agent import DQNAgent
from chess_env import ChessEnv
import sys
import os

class ChessCLI:
    def __init__(self):
        self.board = chess.Board()
        self.agent = None
        self.model_loaded = False
        
    def load_model(self, model_path="final_model.pth"):
        """Load the AI model"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file '{model_path}' not found!")
                return False
                
            print(f"🔄 Loading model: {model_path}")
            self.agent = DQNAgent()
            self.agent.load(model_path)
            self.agent.epsilon = 0.0  # No random moves for best play
            self.model_loaded = True
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def display_board(self):
        """Display the current board position"""
        print("\n" + "="*50)
        print("   a   b   c   d   e   f   g   h")
        print(" +---+---+---+---+---+---+---+---+")
        
        for rank in range(7, -1, -1):
            print(f"{rank+1}|", end="")
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece:
                    print(f" {piece.symbol()} |", end="")
                else:
                    print("   |", end="")
            print(f" {rank+1}")
            print(" +---+---+---+---+---+---+---+---+")
        
        print("   a   b   c   d   e   f   g   h")
        print("="*50)
        
        # Show game status
        if self.board.is_check():
            print("⚠️  CHECK!")
        if self.board.is_checkmate():
            print("🏁 CHECKMATE!")
        elif self.board.is_stalemate():
            print("🤝 STALEMATE!")
        elif self.board.is_insufficient_material():
            print("🤝 INSUFFICIENT MATERIAL!")
        elif self.board.is_seventyfive_moves():
            print("🤝 75-MOVE RULE!")
        elif self.board.is_fivefold_repetition():
            print("🤝 FIVEFOLD REPETITION!")
    
    def get_player_move(self):
        """Get move from human player"""
        while True:
            try:
                move_str = input("\n🎯 Your move (UCI format, e.g., 'e2e4' or 'quit'): ").strip()
                
                if move_str.lower() in ['quit', 'exit', 'q']:
                    return None
                    
                if move_str.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                    
                if move_str.lower() in ['undo', 'u']:
                    if len(self.board.move_stack) > 0:
                        self.board.pop()
                        print("↩️  Move undone!")
                        self.display_board()
                        continue
                    else:
                        print("❌ No moves to undo!")
                        continue
                
                # Parse move
                move = chess.Move.from_uci(move_str)
                
                if move in self.board.legal_moves:
                    return move
                else:
                    print("❌ Illegal move! Try again.")
                    
            except ValueError:
                print("❌ Invalid move format! Use UCI notation (e.g., 'e2e4')")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return None
    
    def show_help(self):
        """Show help information"""
        print("\n📖 HELP:")
        print("  • Enter moves in UCI format: 'e2e4', 'g1f3', 'e7e5', etc.")
        print("  • Type 'undo' or 'u' to take back your last move")
        print("  • Type 'quit' or 'q' to exit the game")
        print("  • Type 'help' or 'h' to show this help")
        print("  • You play as White (uppercase pieces)")
        print("  • AI plays as Black (lowercase pieces)")
    
    def ai_move(self):
        """Get move from AI"""
        if not self.model_loaded:
            print("❌ AI model not loaded!")
            return None
            
        try:
            print("🤖 AI is thinking...")
            
            # Create environment for AI
            env = ChessEnv()
            env.board = self.board.copy()
            
            # Get AI move
            move = self.agent.select_move(env)
            
            if move and move in self.board.legal_moves:
                return move
            else:
                print("❌ AI has no legal moves!")
                return None
                
        except Exception as e:
            print(f"❌ AI error: {e}")
            return None
    
    def play_game(self):
        """Main game loop"""
        print("♟️  ChessBot - Play Against AI")
        print("="*50)
        
        # Load model - dynamically find all .pth files
        import glob
        available_models = []
        
        # Find all .pth files in current directory
        pth_files = glob.glob("*.pth")
        # Also check Model* subdirectories
        pth_files.extend(glob.glob("Model*/*.pth"))
        
        # Sort by name, prioritizing final_model.pth
        pth_files.sort(key=lambda x: (x != "final_model.pth", x))
        
        print("Available models:")
        for i, model in enumerate(pth_files, 1):
            print(f"  {i}. {model}")
            available_models.append(model)
        
        if not available_models:
            print("❌ No .pth model files found!")
            print("💡 Train a model first with: python main.py --games 1000")
            return
            
        while not self.model_loaded:
            try:
                choice = input(f"\nSelect model (1-{len(available_models)}) or enter path: ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                    model_path = available_models[int(choice) - 1]
                else:
                    model_path = choice
                    
                if self.load_model(model_path):
                    break
            except (ValueError, IndexError):
                print("❌ Invalid choice!")
        
        # Start game
        print("\n🎮 Game started! You are White, AI is Black.")
        print("Type 'help' for commands.")
        
        self.display_board()
        
        while not self.board.is_game_over():
            if self.board.turn == chess.WHITE:  # Player's turn
                move = self.get_player_move()
                if move is None:  # Player quit
                    return
            else:  # AI's turn
                move = self.ai_move()
                if move is None:  # AI error
                    return
                    
            # Make the move
            self.board.push(move)
            
            # Show the move
            if self.board.turn == chess.BLACK:  # Just made white move
                print(f"✅ You played: {move.uci()}")
            else:  # Just made black move
                print(f"🤖 AI played: {move.uci()}")
            
            self.display_board()
        
        # Game over
        result = self.board.result()
        print("\n🏁 GAME OVER!")
        
        if result == "1-0":
            print("🎉 Congratulations! You won!")
        elif result == "0-1":
            print("🤖 AI won! Better luck next time.")
        else:
            print("🤝 It's a draw!")
        
        print(f"Final position: {result}")

def main():
    """Main function"""
    try:
        game = ChessCLI()
        game.play_game()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
