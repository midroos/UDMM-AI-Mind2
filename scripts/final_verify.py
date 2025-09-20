import multiprocessing
import os
import sys
import time
import requests
import uvicorn

# Add the src directory to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.udmm2.api.app import app

def run_server():
    """Function to run the uvicorn server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def main():
    """
    Starts the server in a background process, sends a request to it,
    prints the response, and then terminates the server.
    """
    print("Starting server in a background process...")
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    print("Waiting for server to initialize (10s)...")
    time.sleep(10)

    # --- Test /status Endpoint ---
    print("\nSending test query to /status endpoint...")

    try:
        response = requests.get("http://127.0.0.1:8000/status", timeout=15)
        response.raise_for_status()
        data = response.json()

        print("\n--- Full Response ---")
        print(data)
        print("\n--- Analysis ---")

        if "time_step" in data and "system_health" in data:
             print("\nSUCCESS: The /status endpoint returned a valid response.")
        else:
            print("\nFAILURE: The /status endpoint response was not in the expected format.")

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to connect to the server: {e}")

    finally:
        # --- Shutdown ---
        print("\nShutting down server...")
        server_process.terminate()
        server_process.join()
        print("Server process terminated.")

if __name__ == "__main__":
    main()
