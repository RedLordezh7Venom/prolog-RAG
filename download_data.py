from datasets import load_dataset
import json

def download_finqa():
    print("Downloading FinQA dataset (subset)...")
    try:
        # Loading the requested split
        # FinQA requires trust_remote_code=True for its loading script
        dataset = load_dataset("ibm/finqa", split="train[:100]", trust_remote_code=True)
        
        print(f"Successfully loaded {len(dataset)} examples from FinQA train[:100].")
        
        # Look at the first example
        first_example = dataset[0]
        
        print("\n--- First Example Structure ---")
        print(json.dumps(first_example, indent=2))
        
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_finqa()
