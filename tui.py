import chess
import time
import threading
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from agent import DQNAgent
from train import Trainer
from chess_env import ChessEnv

class ChessTUI:
    """Terminal UI for chess RL training"""
    
    def __init__(self):
        self.console = Console()
        self.agent = DQNAgent()
        self.trainer = Trainer(self.agent)
        self.training = False
        self.current_board = None
        
    def render_board(self, board_str: str) -> Panel:
        """Render chess board"""
        return Panel(
            Text(board_str, style="bold white"),
            title="Chess Board",
            border_style="blue"
        )
    
    def render_stats(self, stats: dict) -> Table:
        """Render training statistics"""
        table = Table(title="Training Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        games = stats['games_played']
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        
        win_rate = (wins / games * 100) if games > 0 else 0
        
        table.add_row("Games Played", str(games))
        table.add_row("Wins", str(wins))
        table.add_row("Losses", str(losses))
        table.add_row("Draws", str(draws))
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Epsilon", f"{self.agent.epsilon:.4f}")
        table.add_row("Total Reward", str(stats['total_reward']))
        
        return table
    
    def render_controls(self) -> Panel:
        """Render control instructions"""
        controls = """
[bold cyan]Controls:[/bold cyan]
  s - Start/Stop Training
  q - Quit
  r - Reset Statistics
  1-9 - Set Engine Difficulty
        """
        return Panel(controls, title="Controls", border_style="yellow")
    
    def generate_display(self) -> Layout:
        """Generate full TUI layout"""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=10)
        )
        
        layout["header"].update(
            Panel("[bold magenta]Chess RL Training System[/bold magenta]", 
                  style="bold white on blue")
        )
        
        layout["body"].split_row(
            Layout(self.render_board(str(self.current_board or chess.Board())), 
                   name="board"),
            Layout(self.render_stats(self.trainer.stats), name="stats")
        )
        
        layout["footer"].update(self.render_controls())
        
        return layout
    
    def training_callback(self, game_num, game_result, stats):
        """Callback for training updates"""
        # This will trigger display update
        pass
    
    def train_thread(self, num_games=1000):
        """Training thread"""
        self.trainer.train(
            num_games=num_games,
            callback=self.training_callback
        )
        self.training = False
    
    def run(self):
        """Main TUI loop"""
        env = ChessEnv()
        self.current_board = env.board
        
        self.console.clear()
        self.console.print("[bold green]Chess RL Training System Initialized[/bold green]")
        self.console.print("Press 's' to start training, 'q' to quit\n")
        
        with Live(self.generate_display(), refresh_per_second=2) as live:
            while True:
                try:
                    # Simple display update loop
                    live.update(self.generate_display())
                    time.sleep(0.5)
                    
                    # In a real implementation, you'd handle keyboard input here
                    # For now, auto-start training
                    if not self.training:
                        self.training = True
                        thread = threading.Thread(
                            target=self.train_thread, 
                            args=(100,)
                        )
                        thread.daemon = True
                        thread.start()
                    
                except KeyboardInterrupt:
                    break
        
        self.console.print("\n[bold red]Training stopped[/bold red]")


if __name__ == "__main__":
    tui = ChessTUI()
    tui.run()
