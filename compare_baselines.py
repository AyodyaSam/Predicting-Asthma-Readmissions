import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from xgboost import XGBClassifier

class TraditionalMLBaselines:
    """
    Traditional machine learning baselines using same features as AsthmaGNN
    """
    
    def __init__(self, predictor):
        """
        Initialise from  AsthmaGNN predictor to ensure fair comparison
        """
        self.device = predictor.device
        
        # Extract the same patient features used by AsthmaGNN
        self.patient_features = predictor.graph['patient'].x.cpu().numpy()
        self.labels = predictor.labels.cpu().numpy()
        self.train_mask = predictor.train_mask.cpu().numpy()
        self.val_mask = predictor.val_mask.cpu().numpy()
        self.test_mask = predictor.test_mask.cpu().numpy()
        
        print(f"Traditional ML Baseline Setup:")
        print(f"  Patient features shape: {self.patient_features.shape}")
        print(f"  Total patients: {len(self.labels)}")
        print(f"  Positive rate: {np.mean(self.labels):.1%}")
        print(f"  Train/Val/Test split: {np.sum(self.train_mask)}/{np.sum(self.val_mask)}/{np.sum(self.test_mask)}")
    
    def run_all_baselines(self):
        """Run all traditional ML baselines"""
        
        print("\n" + "="*60)
        print("TRADITIONAL ML BASELINE RESULTS")
        print("="*60)
        
        # Define baseline models
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', 
                random_state=42,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                max_depth=10
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=42,
                scale_pos_weight=len(self.labels[self.labels==0]) / len(self.labels[self.labels==1])
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6
            ),
            'MLP (Neural Network)': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                random_state=42,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15
            ),
            'SVM (RBF)': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42,
                gamma='scale'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(
                    self.patient_features[self.train_mask], 
                    self.labels[self.train_mask]
                )
                
                # Get test predictions
                if hasattr(model, 'predict_proba'):
                    test_probs = model.predict_proba(self.patient_features[self.test_mask])[:, 1]
                else:
                    test_probs = model.decision_function(self.patient_features[self.test_mask])
                
                # Calculate metrics
                test_labels = self.labels[self.test_mask]
                
                if len(np.unique(test_labels)) > 1:
                    auc = roc_auc_score(test_labels, test_probs)
                    
                    # Find optimal threshold
                    precision, recall, thresholds = precision_recall_curve(test_labels, test_probs)
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    best_threshold = thresholds[np.argmax(f1_scores)]
                else:
                    auc = 0.5
                    best_threshold = 0.5
                
                # Binary predictions with optimal threshold
                binary_preds = (test_probs > best_threshold).astype(int)
                accuracy = (binary_preds == test_labels).mean()
                
                # Calculate precision, recall, F1
                tp = np.sum((binary_preds == 1) & (test_labels == 1))
                fp = np.sum((binary_preds == 1) & (test_labels == 0))
                fn = np.sum((binary_preds == 0) & (test_labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                results[model_name] = {
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': best_threshold,
                    'n_positive_pred': np.sum(binary_preds),
                    'n_positive_true': np.sum(test_labels),
                    'n_true_positive': tp
                }
                
                print(f"  AUC: {auc:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                results[model_name] = {'auc': 0.0, 'error': str(e)}
        
        return results
    
    def compare_with_asthmagnn(self, gnn_results):
        """Compare traditional baselines with AsthmaGNN results"""
        
        baseline_results = self.run_all_baselines()
        
        print("\n" + "="*80)
        print("BASELINE vs AsthmaGNN COMPARISON")
        print("="*80)
        
        # Print comparison table
        print(f"{'Model':<25} {'AUC':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Improvement'}")
        print("-" * 80)
        
        # Sort baselines by AUC
        sorted_baselines = sorted(baseline_results.items(), key=lambda x: x[1].get('auc', 0), reverse=True)
        
        best_baseline_auc = 0
        for model_name, metrics in sorted_baselines:
            if 'error' not in metrics:
                auc = metrics['auc']
                f1 = metrics['f1']
                precision = metrics['precision']
                recall = metrics['recall']
                
                print(f"{model_name:<25} {auc:<8.3f} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f}")
                best_baseline_auc = max(best_baseline_auc, auc)
        
        # AsthmaGNN results
        mygnn_auc = gnn_results.get('auc', 0)
        mygnn_f1 = gnn_results.get('f1', 0)
        mygnn_precision = gnn_results.get('precision', 0)
        mygnn_recall = gnn_results.get('recall', 0)
        
        improvement = mygnn_auc - best_baseline_auc
        
        print("-" * 80)
        print(f"{'AsthmaGNN (Ours)':<25} {mygnn_auc:<8.3f} {mygnn_f1:<8.3f} {mygnn_precision:<10.3f} {mygnn_recall:<8.3f} +{improvement:.3f}")
        
        print(f"   Best Traditional ML: {best_baseline_auc:.3f} AUC")
        print(f"   AsthmaGNN: {mygnn_auc:.3f} AUC")
        print(f"   Improvement: +{improvement:.3f} AUC ({(improvement/best_baseline_auc)*100:.1f}% relative gain)")
        
        return {
            'baseline_results': baseline_results,
            'best_baseline_auc': best_baseline_auc,
            'mygnn_auc': mygnn_auc,
            'improvement': improvement
        }

def run_baseline_study(predictor, gnn_test_results):
    """
    Run comprehensive baseline study and comparison
    
    Args:
        predictor: Trained AsthmaGNN predictor
        gnn_test_results: Test results from AsthmaGNN model
    """
    
    print("Starting comprehensive baseline comparison study...")

    baseline_evaluator = TraditionalMLBaselines(predictor)
    comparison_results = baseline_evaluator.compare_with_asthmagnn(gnn_test_results)

    baseline_df = []
    for model_name, metrics in comparison_results['baseline_results'].items():
        if 'error' not in metrics:
            baseline_df.append({
                'model_type': 'Traditional ML',
                'model_name': model_name,
                **metrics
            })

    baseline_df.append({
        'model_type': 'Graph Neural Network',
        'model_name': 'AsthmaGNN',
        **gnn_test_results
    })
    
    baseline_comparison_df = pd.DataFrame(baseline_df)
    baseline_comparison_df.to_csv('baseline_comparison_results.csv', index=False)
    
    print("\nResults saved to: baseline_comparison_results.csv")
