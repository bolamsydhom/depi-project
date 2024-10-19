#%% imports 
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                             precision_score, recall_score, roc_auc_score)
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import os

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify tracking server
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5002')
mlflow.set_tracking_uri(mlflow_tracking_uri)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    all_preds, all_labels = [], []
    running_loss, step = 0.0, 0
    model.train()
    
    for batch in tqdm(train_loader):
        batch.to(device)  
        optimizer.zero_grad() 
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step

def test(epoch, model, test_loader, loss_fn):
    all_preds, all_labels = [], []
    running_loss, step = 0.0, 0
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)  
            pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch) 
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            running_loss += loss.item()
            step += 1
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    log_conf_matrix(all_preds, all_labels, epoch)
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss / step

def log_conf_matrix(y_pred, y_true, epoch):
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    plt.savefig(f'data/images/cm_{epoch}.png')
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\nConfusion matrix: \n{confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except ValueError:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print("ROC AUC: not defined")

# %% Run the training
from mango import scheduler, Tuner
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE

def run_one_training(params):
    params = params[0]
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Load the dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
        test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
        
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
        params["feature_size"] = train_dataset[0].x.shape[1]

        # Prepare training data loaders
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

        # Load the model
        print("Loading model...")
        model_params = {
            'n_layers': 3,
            'embedding_size': 128,
            'n_heads': 4,
            'dropout_rate': 0.1,
            'edge_dim': params["model_edge_dim"],
            'top_k_every_n': 2,
            'top_k_ratio': 0.5,
            'dense_neurons': 256,
            'feature_size': params["feature_size"]
        }
        model = GNN(model_params=model_params).to(device)
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(), 
                                     lr=params["learning_rate"],
                                     momentum=params["sgd_momentum"],
                                     weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
        
        # Start training
        best_loss = float('inf')
        early_stopping_counter = 0
        for epoch in range(300): 
            if early_stopping_counter <= 10: 
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    
                    if float(loss) < best_loss:
                        best_loss = loss
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                break
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]

# %% Hyperparameter search
print("Running hyperparameter search...")
config = {
    "optimizer": "Bayesian",
    "num_iteration": 100
}

tuner = Tuner(HYPERPARAMETERS, 
              objective=run_one_training,
              conf_dict=config) 
results = tuner.minimize()
