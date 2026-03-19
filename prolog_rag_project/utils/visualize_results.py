import json
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_results(results_file="arena_results.json", output_dir="assets"):
    """
    Visualizes RAG benchmarking results (Proof Traces and Average Time).
    """
    if not os.path.exists(results_file):
        print(f"Results file '{results_file}' not found. Run arena.py first.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    systems = ["Prolog-RAG", "Naive", "Graph"]
    proof_counts = {s: 0 for s in systems}
    total_times = {s: [] for s in systems}
    
    for q in data:
        # Some systems might have different names in different versions
        # Mapping standard names to internal ones
        mapping = {
            "Prolog-RAG": "Prolog-RAG",
            "Naive": "Naive",
            "Graph": "Graph"
        }
        
        for sys_name in systems:
            # Check for the key in the answers dict
            ans = q["answers"].get(sys_name)
            if not ans:
                continue
            
            # Count proof traces (has_proof=True OR method='PROLOG')
            if ans.get("has_proof") or ans.get("method") == "PROLOG":
                proof_counts[sys_name] += 1
            
            # Collect query times (always track in ms for consistency)
            t_sec = ans.get("time_sec", 0)
            total_times[sys_name].append(t_sec * 1000)

    # Calculate average times
    avg_times_ms = {s: (sum(total_times[s]) / len(total_times[s])) if total_times[s] else 0 for s in systems}

    # Print Summary Results to Terminal
    print("\nREPORT SUMMARY:")
    print("-" * 30)
    for sys in systems:
        print(f"{sys:<15} | Proof Traces: {proof_counts[sys]} | Avg Time: {avg_times_ms[sys]:.1f}ms")
    print("-" * 30)

    # --- PLOT 1: Proof Trace Availability ---
    plt.figure(figsize=(10, 6))
    colors = ['#27ae60' if s == "Prolog-RAG" else '#bdc3c7' for s in systems]
    bars = plt.bar(systems, [proof_counts[s] for s in systems], color=colors)
    
    plt.title("Proof Trace Availability across RAG Systems", fontweight='bold', fontsize=14)
    plt.ylabel("Number of Successful Proofs", fontsize=12)
    plt.ylim(0, len(data) + 1)
    
    # Annotate bar values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(output_dir, "comparison_proof.png"), dpi=300, bbox_inches='tight')
    plt.close() # Close current figure to start fresh

    # --- PLOT 2: Average Query Latency ---
    plt.figure(figsize=(10, 6))
    time_colors = ['#3498db', '#e74c3c', '#f1c40f'] # Blue, Red, Yellow
    bars = plt.bar(systems, [avg_times_ms[s] for s in systems], color=time_colors)
    
    plt.title("Average Query Latency (ms) across RAG Systems", fontweight='bold', fontsize=14)
    plt.ylabel("Time in Milliseconds", fontsize=12)
    
    # Annotate bar values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 50, f"{int(yval)}ms", ha='center', va='bottom', fontsize=12)

    plt.savefig(os.path.join(output_dir, "comparison_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully saved charts to {output_dir}/")

if __name__ == "__main__":
    visualize_results()
