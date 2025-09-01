import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import pandas as pd

import json
from datetime import datetime


class AsthmaGNN(nn.Module):

    def __init__(self, graph,hidden_dim=64,num_layers=2,dropout=0.3):
        super().__init__()

        self.node_types = graph.node_types
        self.edge_types = graph.edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = dropout

        self.node_projections = nn.ModuleDict()
        for node_type in self.node_types:
            # print(node_type)
            input_dim = graph[node_type].x.shape[1]
            # print(input_dim)
            self.node_projections[node_type] = nn.Linear(input_dim, hidden_dim)


        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict))

        self.classifier = nn.Linear(hidden_dim*3,1)
        # self.classifier = nn.Linear(hidden_dim,1)

        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        """_summary_

        Args:
            x_dict (Dict): Node features {node: features}
            edge_index_dict (Dict): Edge indices {edge_type: edge index}
        """

        # Apply self.node_projections[node_type](x) for each node type
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_projections[node_type](x)

        # Loop through self.convs, apply each HeteroConv layer
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply activation only to intermediate layers
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict_new.items()}
            else:
                x_dict = x_dict_new
                
        # Get patient embeddings, apply classifier
        # print(x_dict.keys())
        patient_embeddings = x_dict['patient']
        med_embeddings = torch.mean(x_dict['medication'], dim=0, keepdim=True)
        comorb_embeddings = torch.mean(x_dict['comorbidity'], dim=0, keepdim=True) 

        combined = torch.cat([
            patient_embeddings,
            med_embeddings.repeat(patient_embeddings.shape[0], 1),
            comorb_embeddings.repeat(patient_embeddings.shape[0], 1)
        ], dim=1)
        
        if self.drop > 0:
            combined = self.dropout(combined)
            # patient_embeddings = self.dropout(patient_embeddings)

        predictions = self.classifier(combined)


        return predictions

    



class AsthmaAdmissionPredictor:
    "Pipeline for training the admission prediction model"

    def __init__(self, graph):
        self.graph = graph['graph_object']
        self.patient_data = graph['patient_data']
        self.readmission_labels = graph.get('readmission_labels', None)
        self.node_mappings = graph['node_mappings']
        self.model = None
        self.results_log = []
        self.model_name = "baseline_gnn"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_descr = ""

        # Track hyperparameters automatically
        self.hyperparams = {
            'hidden_dim': None,
            'num_layers': None,
            'dropout':None,
            'learning_rate': None,
            'epochs': None,
            'batch_size': None,
            'optimizer': None,
            'pos_weight': None
        }




    def set_model_name(self, name):
        """Set a name for this model run"""
        self.model_name = name

    def set_model_description(self, description):
        """Set a description for this model run"""
        self.model_descr=description

    def generate_auto_model_name(self):
        """Generate an automatic model name based on hyperparameters"""
        # print(f"h{self.hyperparams['hidden_dim']}_"
        #            f"l{self.hyperparams['num_layers']}_"
        #            f"lr{self.hyperparams['learning_rate']}")
        if all(v is not None for v in [self.hyperparams['hidden_dim'], 
                                      self.hyperparams['num_layers'], 
                                      self.hyperparams['learning_rate'],
                                      self.hyperparams['dropout']]):
            return (f"h{self.hyperparams['hidden_dim']}_"
                   f"l{self.hyperparams['num_layers']}_"
                   f"lr{self.hyperparams['learning_rate']}_"
                    f"drop{self.hyperparams['dropout']}_"
                    f"{self.model_descr}")
        return "auto_generated"
    

    def create_labels(self):
        "create labels for "
        if self.readmission_labels is None:
            print("No readmission labels found in graph_data!")
            print("Creating placeholder labels...")
            num_patients = len(self.patient_data)
            labels = np.random.binomial(1, 0.3, num_patients)
            print(f"Created placeholder labels: {np.sum(labels)} positive cases out of {len(labels)}")
            return torch.FloatTensor(labels)

        labels = self.align_labels_to_graph()

        print(f"Loaded readmission labels: {np.sum(labels)} positive cases our of {len(labels)}")
        print(f"Readmission rate: {np.mean(labels):2%}")

        return torch.FloatTensor(labels)


    def align_labels_to_graph(self):
        "get patient order and align the labels to the graph"

        pat_order = sorted(self.node_mappings['patient'].items(),key=lambda x: x[1])

        labels = []

        for pat_id, local_id in pat_order:

            pat_readmission = self.readmission_labels[
                self.readmission_labels['SUBJECT_ID'] == pat_id
            ]

            if len(pat_readmission) > 0:
                readmitted = pat_readmission['READMITTED'].iloc[0]
                labels.append(int(readmitted))
            else:
                labels.append(0)

        return np.array(labels)


    def prepare_data(self):
        "prepare data for training"

        self.labels = self.create_labels()

        num_patients = self.graph['patient'].x.shape[0]
        patient_indices = torch.arange(num_patients)

        train_idx, temp_idx = train_test_split(
            patient_indices, test_size=0.4, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42
        )

        self.train_mask = torch.zeros(num_patients, dtype=torch.bool)
        self.val_mask = torch.zeros(num_patients, dtype=torch.bool)
        self.test_mask = torch.zeros(num_patients, dtype=torch.bool)

        self.train_mask[train_idx] = True
        self.val_mask[val_idx] = True
        self.test_mask[test_idx] = True


        self.graph = self.graph.to(self.device)
        self.labels = self.labels.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)


    def create_model(self, hidden_dim=64,num_layers=2,dropout=0.3):

        # print(self.node_mappings)

        # input_dims = {node_type: len(dim) for node_type, dim in self.node_mappings.items()}
        
        # print(input_dims)
        self.hyperparams['hidden_dim'] = hidden_dim
        self.hyperparams['num_layers'] = num_layers
        self.hyperparams['dropout'] = dropout

        self.model = AsthmaGNN(
            graph=self.graph,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device) 


        print(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")

        print(self.model)


    def train_epoch(self, optimizer, criterion):
        """
        Train for one epoch
        """

        self.model.train()
        optimizer.zero_grad()


        # forward pass through entire graph
        predictions = self.model(self.graph.x_dict, self.graph.edge_index_dict)

        # compute loss for training patients
        train_preds = predictions[self.train_mask].squeeze()
        train_labels = self.labels[self.train_mask]
        loss = criterion(train_preds, train_labels)

        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(self, mask):

        self.model.eval()

        with torch.no_grad():
            # get predictions for all patients
            predictions = self.model(self.graph.x_dict, self.graph.edge_index_dict)

            # filter by mask
            mask_preds = predictions[mask].squeeze()
            mask_labels = self.labels[mask]

            # convert to probabilities
            probs = torch.sigmoid(mask_preds)

            # convert to numpy for sklearn metrics
            probs_np = probs.cpu().numpy()
            labels_np = mask_labels.cpu().numpy()

            if len(np.unique(labels_np) > 1):
                auc = roc_auc_score(labels_np, probs_np)
            else:
                print("random prob as only one class")
                auc = 0.5

            binary_preds = (probs_np > 0.5).astype(int)
            accuracy = (binary_preds == labels_np).mean()

            # print(f"Raw scores range: {mask_preds.min():.3f} to {mask_preds.max():.3f}")
            # print(f"Probability range: {probs.min():.3f} to {probs.max():.3f}")
            # print(f"Predictions > 0.5: {(probs > 0.5).sum()}")
            # print(f"Predictions > 0.3: {(probs > 0.3).sum()}")
            # print(f"Predictions > 0.1: {(probs > 0.1).sum()}")

            n_pos_pred = np.sum(binary_preds)
            n_pos_true = np.sum(labels_np)
            n_true_pos = np.sum((binary_preds == 1) & (labels_np == 1))     

            precision = n_true_pos / n_pos_pred if n_pos_pred > 0 else 0
            recall = n_true_pos / n_pos_true if n_pos_true > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    

            return {
                'auc':auc,
                'accuracy':accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_positive_pred':n_pos_pred,
                'n_positive_true':n_pos_true,
                'n_true_positive':n_true_pos,
                'total':len(labels_np)
            }
        

    def train(self, epochs=100, lr=0.01, optimiser_type='adam'):

        self.hyperparams['learning_rate'] = lr
        self.hyperparams['epochs'] = epochs
        self.hyperparams['optimizer'] = optimiser_type

        if self.model_name == "baseline_gnn":
            self.model_name = self.generate_auto_model_name()
            print(f"Using auto-generated model name: {self.model_name}")

        n_positive = self.labels.sum().item()
        n_negative = len(self.labels) - n_positive
        pos_weight = torch.tensor([n_negative / n_positive]).to(self.device)

        print(f"Positive samples: {n_positive}, Negative: {n_negative}")
        print(f"Using pos_weight: {pos_weight.item():.2f}")

        self.hyperparams['pos_weight'] = pos_weight.item()

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if optimiser_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimiser_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimiser_type}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        print("Starting training...")
        print("-" * 50)

        for epoch in range(epochs):

            train_loss = self.train_epoch(optimizer, criterion)

            # evaluate on validation set
            val_metrics = self.evaluate(self.val_mask)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                train_metrics = self.evaluate(self.train_mask)
                self.log_epoch_results(epoch, train_loss, train_metrics, val_metrics)

                print(f"Epoch {epoch:3d}: "
                      f"Loss: {train_loss:.4f}, "
                      f"Train AUC: {train_metrics['auc']:.3f}, "
                      f"Val AUC: {val_metrics['auc']:.3f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.3f}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(f"\n Training completed. Bst validation AUC: {best_val_auc:.3f}")

        self.save_results_to_csv()

        return best_val_auc



    def test(self):
        """Final eval on test set
        """

        print("\n"+"+"*50)
        print("FINAL TEST RESULTS")
        print("\n"+"+"*50)

        train_metrics = self.evaluate(self.train_mask)
        val_metrics = self.evaluate(self.val_mask)
        test_metrics = self.evaluate(self.test_mask)

        print(f"Train AUC: {train_metrics['auc']:.3f}, Accuracy: {train_metrics['accuracy']:.3f}")
        print(f"Val   AUC: {val_metrics['auc']:.3f}, Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"Test  AUC: {test_metrics['auc']:.3f}, Accuracy: {test_metrics['accuracy']:.3f}")


        print("\nTest Set Details:")
        print(f"    Total patients: {test_metrics['total']}")
        print(f"    True readmissions: {test_metrics['n_positive_true']}")
        print(f"    Predicted readmissions: {test_metrics['n_positive_pred']}")
        print(f"    Correct predictions: {test_metrics['n_true_positive']}")

        if test_metrics['n_positive_pred'] > 0:
            precision = test_metrics['n_true_positive']/test_metrics['n_positive_pred']
            print(f"    Precision: {precision:.3f}")

        if test_metrics['n_positive_true'] > 0:
            recall = test_metrics['n_true_positive']/test_metrics['n_positive_true']
            print(f"    Recall: {recall:.3f}")

        self.save_final_test_results(test_metrics)

        return test_metrics
    

    def log_epoch_results(self, epoch, train_loss, train_metrics, val_metrics):
        """Log results for each epoch"""
        log_entry = {
            'model_name': self.model_name,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),

            # Training metrics
            'train_loss': train_loss,
            'train_auc': train_metrics['auc'],
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics.get('f1', 0),
            'train_precision': train_metrics.get('precision', 0),
            'train_recall': train_metrics.get('recall', 0),

            # Validation metrics
            'val_auc': val_metrics['auc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics.get('f1', 0),
            'val_precision': val_metrics.get('precision', 0),
            'val_recall': val_metrics.get('recall', 0),

            # Hyperparameters 
            'hidden_dim': self.hyperparams['hidden_dim'],
            'num_layers': self.hyperparams['num_layers'],
            'drop_out': self.hyperparams['dropout'],
            'learning_rate': self.hyperparams['learning_rate'],
            'total_epochs': self.hyperparams['epochs'],
            'optimizer_type': self.hyperparams['optimizer'],
            'pos_weight': self.hyperparams['pos_weight'],


            'model_description': self.model_descr,
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()) if self.model else None,
        }
        self.results_log.append(log_entry)

    def save_results_to_csv(self, filename="results/model_results.csv"):
        """Save all logged results to CSV"""
        if self.results_log:
            df = pd.DataFrame(self.results_log)
            
            # append to existing file or create new one
            try:
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                combined_df = df
            
            combined_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")

    def save_final_test_results(self, test_metrics, filename="results/final_results.csv"):
        """Save final test results"""
        # final_result = {
        #     'model_name': self.model_name,
        #     'timestamp': datetime.now().isoformat(),
        #     'test_auc': test_metrics['auc'],
        #     'test_accuracy': test_metrics['accuracy'],
        #     'test_precision': test_metrics.get('precision', 0),
        #     'test_recall': test_metrics.get('recall', 0),
        #     'test_f1': test_metrics.get('f1', 0),
        #     'predicted_positives': test_metrics.get('n_positive_pred', 0),
        #     'true_positives': test_metrics.get('n_positive_true', 0),
        #     'correct_positives': test_metrics.get('n_true_positive', 0),
        #     'total_patients': test_metrics.get('total', 0),
        #     'model_description': self.model_descr
        # }
        final_result = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            
            # Test metrics
            'test_auc': test_metrics['auc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics.get('precision', 0),
            'test_recall': test_metrics.get('recall', 0),
            'test_f1': test_metrics.get('f1', 0),
            'predicted_positives': test_metrics.get('n_positive_pred', 0),
            'true_positives': test_metrics.get('n_positive_true', 0),
            'correct_positives': test_metrics.get('n_true_positive', 0),
            'total_patients': test_metrics.get('total', 0),
            
            # Hyperparameters 
            'hidden_dim': self.hyperparams['hidden_dim'],
            'num_layers': self.hyperparams['num_layers'],
            'learning_rate': self.hyperparams['learning_rate'],
            'drop_out': self.hyperparams['dropout'],
            'total_epochs': self.hyperparams['epochs'],
            'optimizer_type': self.hyperparams['optimizer'],
            'pos_weight': self.hyperparams['pos_weight'],
            
            # Model info
            'model_description': self.model_descr,
            'device': str(self.device),
            'total_params': sum(p.numel() for p in self.model.parameters()) if self.model else None,
        }
        
        # Append to CSV
        try:
            existing_df = pd.read_csv(filename)
            new_df = pd.concat([existing_df, pd.DataFrame([final_result])], ignore_index=True)
        except FileNotFoundError:
            new_df = pd.DataFrame([final_result])
        
        new_df.to_csv(filename, index=False)
        print(f"Final results saved to {filename}")


    def run_experiment(self, hidden_dim=64, num_layers=2, dropout=0.3, epochs=100, lr=0.01, 
                      optimiser_type='adam', model_name=None, description=""):
        """
        Convenience method to run a complete experiment with automatic tracking
        """
        if model_name:
            self.set_model_name(model_name)
        if description:
            self.set_model_description(description)
            
        # Run the full pipeline
        self.prepare_data()
        self.create_model(hidden_dim=hidden_dim, num_layers=num_layers,dropout=dropout)
        
            
        best_auc = self.train(epochs=epochs, lr=lr, optimiser_type=optimiser_type)
        test_metrics = self.test()
        
        return test_metrics


if __name__=="__main__":
    hetero_graph = torch.load('datasets/asthma_hetero_graph_complete_v6.pt',weights_only=False)

    predictor = AsthmaAdmissionPredictor(hetero_graph)

    # param_grid = {
    # 'hidden_dim': [32, 64, 128],
    # 'num_layers': [2, 3, 4],
    # 'dropout': [0.1, 0.3, 0.5],
    # 'learning_rate': [0.0001,0.001, 0.01]
    # }
    param_grid = {
    'hidden_dim': [64, 128],
    'num_layers': [2, 3],
    'dropout': [0,0.1, 0.3, 0.5],
    'learning_rate': [0.0001,0.001]
    }
    
    # for the hyper parameter suite
    import itertools
    for i, params in enumerate(itertools.product(*param_grid.values())):
        hidden_dim, num_layers, dropout, lr = params
        # hidden_dim, num_layers, lr = params
        
        predictor = AsthmaAdmissionPredictor(hetero_graph)

        predictor.run_experiment(
            model_name=f"model_{i:03d}_h{hidden_dim}_numl{num_layers}_lr{lr}_drop{dropout}_extended_combined_emb",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=1000,
            description="extended_combined_emb"
            )
    
    # for just one set of results:
    predictor = AsthmaAdmissionPredictor(hetero_graph)
    predictor.prepare_data()

    predictor.create_model(hidden_dim=64, num_layers=2)

    best_auc = predictor.train(epochs=1000, lr=0.001)

    test_results = predictor.test()

    from compare_baselines import run_baseline_study
    run_baseline_study(predictor=predictor,gnn_test_results=test_results)
