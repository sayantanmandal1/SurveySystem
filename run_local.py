#!/usr/bin/env python3
"""
Local development startup script for AI Surveillance System
Optimized for localhost development
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Start the dashboard for local development"""
    print("🚀 Starting AI Surveillance System - Local Development Mode")
    print("=" * 60)
    
    dashboard_path = Path('src/dashboard/app.py')
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        return 1
    
    try:
        # Start Streamlit with localhost configuration
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            str(dashboard_path),
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false',
            '--server.enableCORS=false',
            '--server.enableXsrfProtection=true'
        ]
        
        print("🌐 Dashboard starting at http://localhost:8501")
        print("📝 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())