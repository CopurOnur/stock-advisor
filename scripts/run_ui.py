#!/usr/bin/env python3
"""
Launch the Stock Advisor Web UI.
"""

import sys
import os
import subprocess

def main():
    """Launch the Streamlit web UI."""
    
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Get the UI module path
    ui_module_path = os.path.join(parent_dir, 'stock_advisor', 'ui', 'web_ui.py')
    
    # Launch streamlit
    try:
        print("Starting Stock Advisor Web UI...")
        print("Open your browser to: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        # Set PYTHONPATH environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = parent_dir
        
        subprocess.run([
            sys.executable, 
            '-m', 
            'streamlit', 
            'run', 
            ui_module_path
        ], env=env)
        
    except KeyboardInterrupt:
        print("\nShutting down Stock Advisor Web UI...")
    except Exception as e:
        print(f"Error launching UI: {e}")
        print("Make sure streamlit is installed: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()