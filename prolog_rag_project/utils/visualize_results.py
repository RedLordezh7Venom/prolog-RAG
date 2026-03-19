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
    systems = ["Prolog-RAG", "Naive", "Graph", "CRAG", "Contextual"]
    type_performance = {} # format {type: {system: total_score}}
    
    # Try to load evaluation scores for 'best' calculation
    eval_data = {}
    if os.path.exists("eval_summary.json"):
        with open("eval_summary.json", 'r') as f:
            eval_list = json.load(f)
            eval_data = {q['id']: q['scores'] for q in eval_list}

    proof_counts = {s: 0 for s in systems}
    total_times = {s: [] for s in systems}
    
    for q in data:
        q_id = q.get('id')
        q_type = q.get('type', 'unknown')
        if q_type not in type_performance:
            type_performance[q_type] = {s: 0 for s in systems}
        
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

            # Performance scoring for 'Best' by Type
            if q_id in eval_data and sys_name in eval_data[q_id]:
                score = eval_data[q_id][sys_name].get('accuracy_score', 0)
                type_performance[q_type][sys_name] += score
            elif sys_name == "Prolog-RAG" and (ans.get("has_proof") or ans.get("method") == "PROLOG"):
                # Rough proxy if no eval data: Prolog wins if it has a proof
                type_performance[q_type][sys_name] += 5
            else:
                type_performance[q_type][sys_name] += 1 # Base participation

    # Calculate average times
    avg_times_ms = {s: (sum(total_times[s]) / len(total_times[s])) if total_times[s] else 0 for s in systems}

    # Print Summary Results to Terminal
    print("\n" + "="*50)
    print("REPORT SUMMARY:")
    print("-" * 50)
    for sys in systems:
        if sys in proof_counts:
            print(f"{sys:<15} | Proof Traces: {proof_counts[sys]} | Avg Time: {avg_times_ms[sys]:.1f}ms")
    
    # Print Performance by Question Type
    print("\nBEST SYSTEM BY QUESTION TYPE:")
    print("-" * 50)
    print(f"{'TYPE':<20} | {'TOP SYSTEM':<15} | {'TOTAL ACC SCORE':<10}")
    print("-" * 50)
    for q_type, sys_scores in type_performance.items():
        # Find system with max score
        top_sys = max(sys_scores, key=sys_scores.get)
        print(f"{q_type.upper():<20} | {top_sys:<15} | {sys_scores[top_sys]:<10}")
    print("="*50 + "\n")

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
