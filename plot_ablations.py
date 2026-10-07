import json
import matplotlib.pyplot as plt

def plot_ablations():
    with open('ablation_results.json', 'r') as f:
        results = json.load(f)
        
    nodes = list(results.keys())
    metrics = list(results[nodes[0]].keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(nodes))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[n][metric] for n in nodes]
        ax.bar([p + width*i for p in x], values, width, label=metric)
        
    ax.set_ylabel('Scores')
    ax.set_title('Ablation Results by Number of Nodes')
    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels(nodes)
    ax.legend(loc='lower right')
    
    plt.savefig('ablation_plot.png')
    print("Saved plot to ablation_plot.png")

if __name__ == '__main__':
    plot_ablations()
