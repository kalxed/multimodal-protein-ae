import argparse
import torch
import pickle
import os
import pandas as pd
import sys
import tqdm
from utils import *
sys.path.append(".")
from model.ESM import *
from model.vgae import *
from model.PAE import *

from sklearn.model_selection import GroupKFold
import xgboost as xgb
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process():
    # Define the script directory and data folder
    vgae_model, pae_model, esm_model, concrete_model = load_models()
    data_folder = './downstreamtasks/data/ProtDD'

    # Attempt to read label.csv
    try:
        df = pd.read_csv(f'{data_folder}/label.csv')
    # Create label.csv
    except:
        with open(f'{data_folder}/amino_enzymes.txt', 'r') as f:
            enzyme_ids = [line.strip() for line in f]

        with open(f'{data_folder}/amino_no_enzymes.txt', 'r') as f:
            non_enzyme_ids = [line.strip() for line in f]

        fold_dataframes = []
        for fold_id in range(10):
            with open(f'{data_folder}/amino_fold_{fold_id}.txt', 'r') as f:
                protein_ids = [line.strip() for line in f]

            labels = [1 if id in enzyme_ids else 0 for id in protein_ids]
            fold_df = pd.DataFrame({'id': protein_ids, 'label': labels, 'fold_id': fold_id})
            fold_dataframes.append(fold_df)

        df = pd.concat(fold_dataframes, ignore_index=True)
        df.to_csv(f'{data_folder}/label.csv', index=False)

    print("Number of samples:", len(df))

    # Initialize empty lists to store multimodal representations
    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []

    # Iterate through the HDF5 files and extract multimodal representations
    for i, hdf5_file in tqdm(enumerate(df['id']), total=len(df['id'])):
        hdf5_path = f'/{data_folder}/hdf5/{hdf5_file}.hdf5'
        
        # Get multimodal representations from the HDF5 file using pre-trained models
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_modalities(hdf5_path, esm_model, vgae_model, pae_model, concrete_model)
        
        # Append the representations to their respective lists
        mulmodal.append(multimodal_representation.detach().numpy())
        sequence.append(encoded_sequence.detach().numpy())
        graph.append(encoded_graph.detach().numpy())
        point_cloud.append(encoded_point_cloud.detach().numpy())

    # Save the multimodal representations to pickle files
    pickle_dump(data_folder, mulmodal, sequence, graph, point_cloud)
    # with open(f'{data_folder}/multimodal.pkl', 'wb') as f:
    #     pickle.dump(mulmodal, f)

    # with open(f'{data_folder}/sequence.pkl', 'wb') as f:
    #     pickle.dump(sequence, f)

    # with open(f'{data_folder}/graph.pkl', 'wb') as f:
    #     pickle.dump(graph, f)

    # with open(f'{data_folder}/point_cloud.pkl', 'wb') as f:
    #     pickle.dump(point_cloud, f)
    

def load_data(modal, data_folder):
    # Load input data (X) from a pickle file and labels (y) from a CSV file
    with open(f'{data_folder}/{modal}.pkl', 'rb') as f:
        tensor_list = pickle.load(f)
        data = [tensor.detach().numpy() for tensor in tensor_list]
        X = np.array(data)

    df = pd.read_csv(f'{data_folder}/label.csv')
    y = df['label']
    fold_ids = df['fold_id']
    
    return X, y, fold_ids

def train(modal):
    data_folder = './downstreamtasks/data/ProtDD'
    # Load data and perform 10-fold cross-validation
    X, y, fold_ids = load_data(modal, data_folder)
    cv = GroupKFold(n_splits=10)

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Split the data into training and testing sets for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize and train the XGBoost classifier
        model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective="binary:logistic")
        model.fit(X_train, y_train, eval_metric='logloss', eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=10, verbose=0)

        # Save the trained model for each fold
        model.save_model(f"{data_folder}{modal}_model/xgb_model_fold{i+1}.json")

def test(modal):
    data_folder = './downstreamtasks/data/ProtDD'
    # Load data and perform 10-fold cross-validation for testing
    X, y, fold_ids = load_data(modal, data_folder)
    cv = GroupKFold(n_splits=10)

    fold_scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Load the trained model for each fold
        X_test, y_test = X[test_index], y.iloc[test_index]
        model = XGBClassifier()
        model.load_model(f"{data_folder}{modal}_model/xgb_model_fold{i+1}.json")

        # Make predictions and calculate accuracy for each fold
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_scores.append(accuracy)

    # Print the mean accuracy across all folds
    print("Mean Accuracy:", np.mean(fold_scores))
