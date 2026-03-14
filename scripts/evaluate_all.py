#!/usr/bin/env python
"""Script to evaluate all models"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from src.training.metrics import plot_confusion_matrix
from src.utils.logger import setup_logger
from src.utils.paths import paths

logger = setup_logger(__name__)

# Model mapping
MODELS = {
    'A': 'memberA_1dcnn',
    'B': 'memberB_2dcnn',
    'C': 'memberC_transformer',
    'D': 'memberD_multimodal',
    'E': 'memberE_baseline'
}


def load_test_data():
    """Load test data"""
    data_path = paths.data_processed
    
    # Load labels and splits
    labels = np.load(data_path / 'labels.npy')
    splits = np.load(data_path / 'splits.npy', allow_pickle=True)
    
    # Get test indices
    test_mask = splits == 'test'
    test_labels = labels[test_mask]
    
    return test_labels, test_mask


def load_model_predictions(member: str, test_mask: np.ndarray) -> np.ndarray:
    """
    Load predictions for a specific member
    
    Args:
        member: Member letter (A, B, C, D, E)
        test_mask: Boolean mask for test samples
    
    Returns:
        Predictions array
    """
    model_dir = Path(f'./results/{MODELS[member]}')
    
    # Try different prediction file names
    pred_files = [
        model_dir / 'predictions.npy',
        model_dir / 'test_predictions.npy',
        model_dir / 'final_predictions.npy'
    ]
    
    for pred_file in pred_files:
        if pred_file.exists():
            predictions = np.load(pred_file)
            # Check if predictions are for all samples or just test
            if len(predictions) == len(test_mask):
                return predictions[test_mask]
            else:
                return predictions
    
    raise FileNotFoundError(f"No predictions found for member {member}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate sleep staging models')
    parser.add_argument('--member', type=str, choices=['A', 'B', 'C', 'D', 'E'],
                       help='Member to evaluate (A, B, C, D, E). If not specified, evaluate all.')
    args = parser.parse_args()
    
    # Determine which members to evaluate
    if args.member:
        members_to_evaluate = [args.member]
        logger.info(f"Evaluating member {args.member} only")
    else:
        members_to_evaluate = list(MODELS.keys())
        logger.info("Evaluating all members")
    
    # Load test data
    test_labels, test_mask = load_test_data()
    
    # Collect all results
    all_results = {}
    all_predictions = {}
    
    for member in members_to_evaluate:
        model_name = MODELS[member]
        logger.info(f"Evaluating member {member}...")
        
        try:
            predictions = load_model_predictions(member, test_mask)
            
            # Compute metrics
            report = classification_report(
                test_labels,
                predictions,
                target_names=['W', 'N1', 'N2', 'N3', 'REM'],
                output_dict=True
            )
            
            all_results[member] = {
                'model': model_name,
                'accuracy': report['accuracy'],
                'f1_macro': report['macro avg']['f1-score'],
                'per_class': {
                    stage: report[stage]['f1-score']
                    for stage in ['W', 'N1', 'N2', 'N3', 'REM']
                }
            }
            
            all_predictions[member] = predictions
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(
                test_labels,
                predictions,
                save_path=paths.results / f'{model_name}/confusion_matrix.png'
            )
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to evaluate member {member}: {e}")
            all_results[member] = {'error': str(e)}
    
    # Create comparison table
    comparison = {
        'member': [],
        'model': [],
        'accuracy': [],
        'f1_macro': [],
        'f1_W': [],
        'f1_N1': [],
        'f1_N2': [],
        'f1_N3': [],
        'f1_REM': []
    }
    
    for member, results in all_results.items():
        if 'error' not in results:
            comparison['member'].append(member)
            comparison['model'].append(results['model'])
            comparison['accuracy'].append(results['accuracy'])
            comparison['f1_macro'].append(results['f1_macro'])
            comparison['f1_W'].append(results['per_class']['W'])
            comparison['f1_N1'].append(results['per_class']['N1'])
            comparison['f1_N2'].append(results['per_class']['N2'])
            comparison['f1_N3'].append(results['per_class']['N3'])
            comparison['f1_REM'].append(results['per_class']['REM'])
    
    # Save comparison
    results_dir = paths.results
    paths.ensure_dir(results_dir)
    
    # Save individual results if evaluating single member
    if args.member:
        individual_results_file = results_dir / f'member{args.member}_results.json'
        with open(individual_results_file, 'w') as f:
            json.dump(all_results[args.member], f, indent=2)
        logger.info(f"Individual results saved to {individual_results_file}")
    
    # Save comparison (only if evaluating all or if comparison makes sense)
    if len(members_to_evaluate) > 1:
        with open(results_dir / 'comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    # Print results
    logger.info("\n" + "="*60)
    if args.member:
        logger.info(f"EVALUATION RESULTS FOR MEMBER {args.member}")
    else:
        logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*60)
    
    if len(comparison['member']) > 0:
        print(f"\n{'Member':<6} {'Model':<20} {'Accuracy':<10} {'F1-Macro':<10}")
        print("-"*50)
        
        for i, member in enumerate(comparison['member']):
            print(f"{member:<6} {comparison['model'][i]:<20} "
                  f"{comparison['accuracy'][i]:<10.4f} "
                  f"{comparison['f1_macro'][i]:<10.4f}")
        
        # Find best model (only if multiple models evaluated)
        if len(members_to_evaluate) > 1:
            best_idx = np.argmax(comparison['f1_macro'])
            logger.info(f"\nBest model: Member {comparison['member'][best_idx]} "
                        f"({comparison['model'][best_idx]}) "
                        f"with F1-macro = {comparison['f1_macro'][best_idx]:.4f}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()