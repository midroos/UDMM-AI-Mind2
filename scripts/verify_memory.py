import multiprocessing
import os
import sys
import time
import requests
import uvicorn
from tqdm import tqdm

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

    # Wait for the server to start
    print("Waiting for server to initialize...")
    time.sleep(10)

    # --- Test Query ---
    question = "ما هي عاصمة فرنسا؟"
    print(f"\nSending test query: '{question}'")

    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        print("\n--- Full Response ---")
        print(data)
        print("\n--- Analysis ---")

        contexts = data.get("contexts", [])
        if not contexts:
            print("ERROR: No contexts were returned.")
        else:
            top_context = contexts[0]
            retrieved_q = top_context.get("meta", {}).get("text")
            retrieved_a = top_context.get("meta", {}).get("answer")
            score = top_context.get("score")

            print(f"Retrieved Question: '{retrieved_q}'")
            print(f"Retrieved Answer:   '{retrieved_a}'")
            print(f"Similarity Score:   {score:.4f}")

            if "باريس" in retrieved_a:
                print("\nSUCCESS: The correct answer was retrieved from the new knowledge base.")
            else:
                print("\nFAILURE: The retrieved answer was not the expected one.")

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
