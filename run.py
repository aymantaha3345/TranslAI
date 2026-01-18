#!/usr/bin/env python3
"""
TRANSLai - Run Script
Correct version for handling reload mode in uvicorn
"""

import os
import sys
import platform
from pathlib import Path
import time

def setup_environment():
    """Setup environment and correct paths"""
    project_root = Path(__file__).parent.resolve()
    os.chdir(project_root)
    
    print(f"✅ Project directory: {project_root}")
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Add project directory to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 TRANSLai - Multilingual Prompt Translation & Enhancement")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup environment
    project_root = setup_environment()
    
    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found")
        print("   Creating .env file from .env.example")
        import shutil
        shutil.copy2(project_root / ".env.example", env_file)
        print("   ✅ .env file created")
    
    try:
        # Run uvicorn with import string (correct solution for reload)
        import uvicorn
        
        print("\n🚀 Starting TRANSLai server...")
        print("🌐 Address: http://0.0.0.0:8000")
        print("📚 API Docs: http://localhost:8000/api/docs")
        print("🔧 Development mode: Auto-reload enabled")
        
        # Use import string instead of importing app object directly
        uvicorn.run(
            "translai.app.main:app",  # ← This is the correct solution
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["translai"],
            log_level="info",
            access_log=False
        )
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n💡 Suggested solutions:")
        print("   1. Check project structure")
        print("   2. Ensure virtual environment is activated")
        print("   3. Try: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Performance info
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Initialization time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped manually")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)