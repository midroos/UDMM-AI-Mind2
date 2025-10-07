import time
print(f"[{time.time()}] Starting diagnosis...")
try:
    from src.udmm2.udmm_core import UDMMAgent
    print(f"[{time.time()}] Import successful. Initializing UDMMAgent...")
    agent = UDMMAgent()
    print(f"[{time.time()}] UDMMAgent initialized successfully.")
except Exception as e:
    print(f"[{time.time()}] An error occurred: {e}")
print(f"[{time.time()}] Diagnosis finished.")