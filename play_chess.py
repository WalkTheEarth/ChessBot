"""
ChessBot - Play Against Your Trained AI
A simple GUI to play chess against your trained neural network
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import chess
import chess.svg
from PIL import Image, ImageTk
import io
import cairosvg
from agent import DQNAgent
from chess_env import ChessEnv
import threading
import time

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ChessBot - Play Against AI")
        self.root.geometry("800x700")
        self.root.configure(bg='#2b2b2b')
        
        # Chess components
        self.board = chess.Board()
        self.agent = None
        self.model_loaded = False
        self.selected_square = None
        self.game_over = False
        
        # GUI setup
        self.setup_gui()
        self.load_default_model()
        
    def setup_gui(self):
        """Setup the GUI components"""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="♟️ ChessBot - Play Against AI", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="AI Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                       state="readonly", width=25)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Populate model list
        self.update_model_list()
        
        # Load model button
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=(0, 5))
        
        # Refresh models button
        ttk.Button(control_frame, text="Refresh", 
                  command=self.update_model_list).pack(side=tk.LEFT, padx=(0, 5))
        
        # Browse for model file button
        ttk.Button(control_frame, text="Browse...", 
                  command=self.browse_model_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # New game button
        ttk.Button(control_frame, text="New Game", 
                  command=self.new_game).pack(side=tk.LEFT, padx=(0, 10))
        
        # AI difficulty (epsilon)
        ttk.Label(control_frame, text="AI Strength:").pack(side=tk.LEFT, padx=(20, 5))
        self.difficulty_var = tk.DoubleVar(value=0.0)
        difficulty_scale = ttk.Scale(control_frame, from_=0.0, to=0.5, 
                                   variable=self.difficulty_var, orient=tk.HORIZONTAL,
                                   length=100)
        difficulty_scale.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(control_frame, text="(0=Best, 0.5=Random)").pack(side=tk.LEFT)
        
        # Game info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(info_frame, text="Loading AI model...", 
                                     font=('Arial', 12))
        self.status_label.pack(side=tk.LEFT)
        
        # Move history
        self.move_history = tk.Text(info_frame, height=3, width=30, 
                                   font=('Courier', 9), bg='#1e1e1e', fg='white')
        self.move_history.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Chess board frame
        board_frame = ttk.Frame(main_frame)
        board_frame.pack(expand=True, fill=tk.BOTH)
        
        # Create chess board
        self.create_chess_board(board_frame)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Click a piece to select it, then click destination square. You play as White, AI plays as Black.",
                               font=('Arial', 10), foreground='gray')
        instructions.pack(pady=(10, 0))
        
    def create_chess_board(self, parent):
        """Create the chess board GUI"""
        self.board_frame = ttk.Frame(parent)
        self.board_frame.pack(expand=True)
        
        # Create 8x8 grid of buttons for the chess board
        self.squares = {}
        for row in range(8):
            for col in range(8):
                square_name = chr(ord('a') + col) + str(8 - row)
                square = tk.Button(self.board_frame, width=6, height=3,
                                 font=('Arial', 20, 'bold'),
                                 command=lambda s=square_name: self.on_square_click(s))
                square.grid(row=row, col=col, padx=1, pady=1)
                self.squares[square_name] = square
                
        self.update_board_display()
        
    def update_board_display(self):
        """Update the visual representation of the chess board"""
        for square_name, button in self.squares.items():
            square = chess.parse_square(square_name)
            piece = self.board.piece_at(square)
            
            if piece:
                # Set piece symbol and color
                piece_symbol = piece.symbol()
                if piece.color == chess.WHITE:
                    button.configure(text=piece_symbol, fg='white', bg='#f0d9b5')
                else:
                    button.configure(text=piece_symbol, fg='black', bg='#b58863')
            else:
                button.configure(text='', bg='#f0d9b5' if (chess.square_file(square) + chess.square_rank(square)) % 2 == 0 else '#b58863')
            
            # Highlight selected square
            if self.selected_square == square_name:
                button.configure(relief=tk.SUNKEN, bd=3)
            else:
                button.configure(relief=tk.RAISED, bd=1)
                
    def on_square_click(self, square_name):
        """Handle square click events"""
        if self.game_over or not self.model_loaded:
            return
            
        if self.board.turn != chess.WHITE:  # Not player's turn
            return
            
        if self.selected_square is None:
            # Select a piece
            square = chess.parse_square(square_name)
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square_name
                self.update_board_display()
        else:
            # Try to make a move
            try:
                move = chess.Move.from_uci(self.selected_square + square_name)
                if move in self.board.legal_moves:
                    self.make_move(move)
                    self.selected_square = None
                    self.update_board_display()
                    
                    # AI's turn
                    if not self.game_over and self.board.turn == chess.BLACK:
                        self.root.after(500, self.ai_move)  # Delay for better UX
                else:
                    # Invalid move, try to select another piece
                    square = chess.parse_square(square_name)
                    piece = self.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        self.selected_square = square_name
                    else:
                        self.selected_square = None
                    self.update_board_display()
            except:
                self.selected_square = None
                self.update_board_display()
                
    def make_move(self, move):
        """Make a move and update the game state"""
        self.board.push(move)
        self.add_move_to_history(move)
        self.check_game_over()
        
    def add_move_to_history(self, move):
        """Add move to the history display"""
        move_str = f"{len(self.board.move_stack)}. {move.uci()}"
        self.move_history.insert(tk.END, move_str + "\n")
        self.move_history.see(tk.END)
        
    def ai_move(self):
        """Make AI move in a separate thread"""
        if self.game_over or not self.model_loaded:
            return
            
        def ai_move_thread():
            try:
                # Set AI difficulty
                if self.agent:
                    self.agent.epsilon = self.difficulty_var.get()
                
                # Create environment for AI
                env = ChessEnv()
                env.board = self.board.copy()
                
                # Get AI move
                move = self.agent.select_move(env)
                
                if move and move in self.board.legal_moves:
                    # Make move on main thread
                    self.root.after(0, lambda: self.make_ai_move(move))
                else:
                    self.root.after(0, lambda: self.status_label.configure(text="AI has no legal moves!"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(text=f"AI Error: {str(e)}"))
        
        # Show AI thinking
        self.status_label.configure(text="AI is thinking...")
        
        # Run AI in separate thread
        thread = threading.Thread(target=ai_move_thread, daemon=True)
        thread.start()
        
    def make_ai_move(self, move):
        """Make AI move on main thread"""
        self.make_move(move)
        self.update_board_display()
        
        if not self.game_over:
            self.status_label.configure(text="Your turn (White)")
        else:
            self.status_label.configure(text="Game Over!")
            
    def check_game_over(self):
        """Check if game is over and update status"""
        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            if result == "1-0":
                messagebox.showinfo("Game Over", "Congratulations! You won!")
                self.status_label.configure(text="You won!")
            elif result == "0-1":
                messagebox.showinfo("Game Over", "AI won! Better luck next time.")
                self.status_label.configure(text="AI won!")
            else:
                messagebox.showinfo("Game Over", "It's a draw!")
                self.status_label.configure(text="Draw!")
        else:
            if self.board.turn == chess.WHITE:
                self.status_label.configure(text="Your turn (White)")
            else:
                self.status_label.configure(text="AI's turn (Black)")
                
    def new_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.selected_square = None
        self.game_over = False
        self.move_history.delete(1.0, tk.END)
        self.update_board_display()
        self.status_label.configure(text="New game started! Your turn (White)")
        
    def update_model_list(self):
        """Update the list of available models"""
        import os
        import glob
        
        # Find all .pth files
        pth_files = glob.glob("*.pth")
        pth_files.extend(glob.glob("Model*/*.pth"))  # Include models in subdirectories
        
        # Sort by name, prioritizing final_model.pth
        pth_files.sort(key=lambda x: (x != "final_model.pth", x))
        
        if pth_files:
            self.model_combo['values'] = pth_files
            self.model_var.set(pth_files[0])  # Set first model as default
        else:
            self.model_combo['values'] = ["No models found"]
            self.model_var.set("No models found")
    
    def load_default_model(self):
        """Load the default model"""
        try:
            if self.model_var.get() != "No models found":
                self.load_model()
        except:
            self.status_label.configure(text="No model loaded. Please select a model file.")
            
    def load_model(self):
        """Load the selected model"""
        model_path = self.model_var.get()
        try:
            self.agent = DQNAgent()
            self.agent.load(model_path)
            self.model_loaded = True
            self.status_label.configure(text=f"Model loaded: {model_path}")
            messagebox.showinfo("Success", f"Successfully loaded {model_path}")
        except Exception as e:
            self.model_loaded = False
            self.status_label.configure(text=f"Failed to load model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            
    def browse_model_file(self):
        """Browse for a model file"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select ChessBot Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.model_var.set(file_path)
            self.load_model()
    
    def on_model_change(self, event):
        """Handle model selection change"""
        if self.model_loaded and self.model_var.get() != "No models found":
            self.load_model()

def main():
    """Main function to run the chess GUI"""
    root = tk.Tk()
    app = ChessGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
