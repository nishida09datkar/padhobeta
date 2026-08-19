import os
import sys

# Vercel serverless: add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force uploads to /tmp (only writable dir on Vercel)
os.environ["UPLOAD_DIR"] = "/tmp/uploads"

from main import app

# Vercel expects the ASGI app as `app`
