Artefex - Neural Forensic Restoration
======================================

Getting Started
---------------
1. Extract this zip to any folder.
2. Double-click Artefex.exe to start.
   - If running from source, double-click Artefex.bat instead.
3. Your browser will open automatically to the Artefex web UI.
4. Drag and drop images to analyze them.
5. Close the terminal window to stop the server.

Troubleshooting
---------------
- If Windows SmartScreen blocks the app, click
  "More info" then "Run anyway".
- If port 8787 is busy, the app will pick the next
  available port automatically.
- Make sure no antivirus software is blocking the app.

Running from Source
-------------------
If you cloned the repository instead of downloading a
built release:

1. Install Python 3.10 or newer from
   https://www.python.org/downloads/
   Make sure to check "Add Python to PATH" during install.

2. Open a terminal in the artefex folder and run:
   pip install -e ".[web]"

3. Double-click Artefex.bat to start, or run:
   python launcher.py

For more information visit:
  https://github.com/turnert2005/artefex
