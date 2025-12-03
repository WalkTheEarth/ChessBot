"""
ChessBot Game Launcher
Easy way to start playing against your trained AI
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import chess
    except ImportError:
        missing.append("python-chess")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (usually comes with Python)")
    
    return missing

def main():
    print("♟️  ChessBot Game Launcher")
    print("=" * 40)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return
    
    # Check for model files
    model_files = [
        "final_model.pth", "checkpoint_500.pth", "checkpoint_400.pth",
        "checkpoint_300.pth", "checkpoint_200.pth", "checkpoint_100.pth"
    ]
    
    available_models = [f for f in model_files if os.path.exists(f)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("💡 Train a model first with: python main.py --games 1000")
        return
    
    print(f"✅ Found {len(available_models)} trained model(s)")
    
    # Choose game mode
    print("\n🎮 Choose game mode:")
    print("1. GUI Version (Recommended)")
    print("2. Command Line Version")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                print("🚀 Launching GUI version...")
                try:
                    subprocess.run([sys.executable, "play_chess.py"], check=True)
                except subprocess.CalledProcessError:
                    print("❌ Failed to launch GUI. Trying command line version...")
                    subprocess.run([sys.executable, "play_chess_cli.py"], check=True)
                break
                
            elif choice == "2":
                print("🚀 Launching command line version...")
                subprocess.run([sys.executable, "play_chess_cli.py"], check=True)
                break
                
            elif choice == "3":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
