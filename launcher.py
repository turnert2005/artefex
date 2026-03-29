"""Artefex launcher - starts the web UI and opens a browser."""

import socket
import sys
import threading
import time
import webbrowser


def find_open_port(start=8787, max_tries=10):
    """Find an open port starting from the given port number."""
    for offset in range(max_tries):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def open_browser(port, delay=1.5):
    """Open the default browser after a short delay."""
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


def main():
    """Start the Artefex web server and open a browser."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed.\n"
            "Install web dependencies with:\n"
            "  pip install artefex[web]\n"
        )
        sys.exit(1)

    try:
        from artefex.web import app  # noqa: F401
    except ImportError:
        print(
            "Error: artefex is not installed.\n"
            "Install with:\n"
            "  pip install -e .[web]\n"
        )
        sys.exit(1)

    port = find_open_port(8787)
    if port is None:
        print("Error: Could not find an open port (tried 8787-8796).")
        sys.exit(1)

    if port != 8787:
        print(f"Port 8787 was in use, using port {port} instead.")

    print()
    print("=" * 52)
    print(f"  Artefex is running at http://localhost:{port}")
    print("  Press Ctrl+C to stop")
    print("=" * 52)
    print()

    # Open browser in a background thread
    browser_thread = threading.Thread(
        target=open_browser, args=(port,), daemon=True
    )
    browser_thread.start()

    try:
        uvicorn.run(
            "artefex.web:app",
            host="127.0.0.1",
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nArtefex stopped.")


if __name__ == "__main__":
    main()
