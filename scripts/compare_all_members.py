import numpy as np
import json
from sklearn.metrics import classification_report

def load_member_results(member):
    try:
        pred = np.load(f'./results/member{member}/predictions.npy')
        true = np.load(f'./results/member{member}/labels.npy')
        return pred, true
    except:
        return None, None

def main():
    members = ['A', 'B', 'C', 'D', 'E']
    all_results = {}
    
    for member in members:
        pred, true = load_member_results(member)
        if pred is not None:
            report = classification_report(true, pred, 
                target_names=['W','N1','N2','N3','REM'],
                output_dict=True)
            all_results[member] = {
                'accuracy': report['accuracy'],
                'f1_macro': report['macro avg']['f1-score']
            }
    
    print("\n" + "="*50)
    print("Model Comparison (All members use same test set)")
    print("="*50)
    print(f"{'Member':<10} {'Accuracy':<15} {'F1-Macro':<15}")
    print("-"*50)
    
    for member, results in all_results.items():
        print(f"{member:<10} {results['accuracy']:<15.4f} {results['f1_macro']:<15.4f}")
    
    with open('./results/comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()