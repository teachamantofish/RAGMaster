# summary_wrapper_hf.py uses a large model on Hugging Face.  
# While you can run 4summary.py directly if you have access to a large model,
# it is slow on a local machine.  This wrapper uses a Hugging Face Inference Endpoint
#
# wrapper_qwen80b_subprocess.py
# Setup steps: 
# pip install huggingface-hub
# run huggingface-cli login
# Find a model in HF
# Choose Deploy -> Inference endpoint -> Create endpoint
# Copy the endpoint url below
import os, sys, subprocess
from huggingface_hub import get_inference_endpoint

# === CONFIG CONSTANTS ===
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
ENDPOINT_NAME = "qwen3-next-80b-a3b-instruct-hmt"    # name of your endpoint on HF
TRANSFORMERS_CACHE = os.path.abspath(".hf_cache")
AUTO_PAUSE_ENDPOINT = True  # Set to True to auto-pause endpoint after each run (saves $ but adds 5+ min startup next time)

# =========================

def main():
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
    env["CHUNK_SUMMARY_MODEL"] = "huggingface:tgi"  # Override to use HuggingFace backend

    # === Resume endpoint before running ===
    try:
        ep = get_inference_endpoint(ENDPOINT_NAME)
        status = ep.status
        print(f"Current endpoint status: {status}")
        
        # If already running, use it
        if status == "running":
            endpoint_url = ep.url
            print(f"Endpoint '{ENDPOINT_NAME}' is already running at {endpoint_url}")
            env["HF_ENDPOINT_URL"] = endpoint_url
        
        # If paused, try to resume
        elif status == "paused":
            print(f"Resuming endpoint '{ENDPOINT_NAME}'...")
            ep.resume()
            print(f"Waiting for endpoint to be ready...")
            ep.wait(timeout=500)  # Wait time in seconds
            
            # Get fresh endpoint info after resume
            ep = get_inference_endpoint(ENDPOINT_NAME)
            endpoint_url = ep.url
            print(f"Endpoint '{ENDPOINT_NAME}' is ready at {endpoint_url}")
            env["HF_ENDPOINT_URL"] = endpoint_url
        
        # If initializing, wait for it to be ready
        elif status == "initializing":
            print(f"Endpoint '{ENDPOINT_NAME}' is initializing. Waiting for it to be ready...")
            ep.wait(timeout=300)  # Wait up to 5 minutes
            
            # Get fresh endpoint info after initialization
            ep = get_inference_endpoint(ENDPOINT_NAME)
            endpoint_url = ep.url
            print(f"Endpoint '{ENDPOINT_NAME}' is ready at {endpoint_url}")
            env["HF_ENDPOINT_URL"] = endpoint_url
        
        # Handle other statuses
        else:
            print(f"ERROR: Endpoint is in unexpected state: {status}")
            print(f"Endpoint details: {ep}")
            raise RuntimeError(f"Cannot use endpoint in '{status}' state. Please check HuggingFace console for details.")
            
    except Exception as e:
        print(f"Error with endpoint: {e}")
        print(f"Please check the endpoint status at: https://ui.endpoints.huggingface.co/")
        raise

    try:
        # === Run your existing summary script as-is ===
        subprocess.run(
            [sys.executable, "4summary.py", *sys.argv[1:]],
            env=env,
            check=True
        )
    finally:
        # Optionally pause the endpoint after use (saves money but adds startup time for next run)
        if AUTO_PAUSE_ENDPOINT:
            try:
                ep = get_inference_endpoint(ENDPOINT_NAME)
                ep.pause()
                print(f"Endpoint '{ENDPOINT_NAME}' paused successfully.")
            except Exception as e:
                print(f"Warning: could not pause endpoint: {e}")
        else:
            print("Auto-pause disabled. Endpoint will remain running.")

if __name__ == "__main__":
    main()