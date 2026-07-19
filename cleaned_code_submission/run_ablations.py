import os
import subprocess
import json
nodes_to_test = [16, 32, 128]
epochs = 50
model = 'unet_gcn'
results = {}
for num_nodes in nodes_to_test:
    print(f'==================================================')
    print(f'Starting Training for num_nodes={num_nodes}')
    print(f'==================================================')
    train_cmd = ['python', 'train.py', '--model', model, '--num_nodes', str(num_nodes), '--epochs', str(epochs)]
    subprocess.run(train_cmd, check=True)
    print(f'==================================================')
    print(f'Starting Evaluation for num_nodes={num_nodes}')
    print(f'==================================================')
    checkpoint_path = f'checkpoints/best_model_{model}_nodes_{num_nodes}.pth'
    eval_cmd = ['python', 'evaluate.py', '--model', model, '--num_nodes', str(num_nodes), '--checkpoint', checkpoint_path, '--save_preds']
    result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    metrics = {}
    for line in result.stdout.split('\n'):
        if line.startswith('Dice:'):
            metrics['Dice'] = float(line.split(':')[1].strip())
        elif line.startswith('IoU:'):
            metrics['IoU'] = float(line.split(':')[1].strip())
        elif line.startswith('Precision:'):
            metrics['Precision'] = float(line.split(':')[1].strip())
        elif line.startswith('Recall:'):
            metrics['Recall'] = float(line.split(':')[1].strip())
    results[num_nodes] = metrics
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
print('Ablation study completed!')
print(json.dumps(results, indent=4))
