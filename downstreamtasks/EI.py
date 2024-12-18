import argparse
import torch
import pickle
import os
import pandas as pd
import sys
from tqdm import tqdm
from utils import *
from mpae.nn.esm import *
from mpae.nn.vgcn import *
from mpae.nn.pae import *
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GroupKFold
import xgboost as xgb
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def process(batches):
    if batches:
        print("Processing in batches is not supported for this task.")
        sys.exit(1)
    
    # Define the script directory and data folder
    vgae_model, pae_model, esm_model, concrete_model = load_models()
    vgae_model = vgae_model.to(device)
    pae_model = pae_model.to(device)
    esm_model = esm_model.to(device)
    concrete_model = concrete_model.to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "..", "data", "ProtDD")

    # Normalize the path (resolve "..")
    data_folder = os.path.abspath(target_dir)

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
        hdf5_path = f'/{data_folder}/data/{hdf5_file}.hdf5'
        
        # Get multimodal representations from the HDF5 file using pre-trained models
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_modalities(hdf5_path, esm_model, vgae_model, pae_model, concrete_model, device=device)
        
        if any(x is None for x in [multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud]):
            continue
        
        # Append the representations to their respective lists
        mulmodal.append(multimodal_representation.detach().cpu().numpy())
        sequence.append(encoded_sequence.detach().cpu().numpy())
        graph.append(encoded_graph.detach().cpu().numpy())
        point_cloud.append(encoded_point_cloud.detach().cpu().numpy())

    # Save the multimodal representations to pickle files
    pickle_dump(data_folder, mulmodal, sequence, graph, point_cloud)
    

def load_data(modal, data_folder, batches):
    if batches:
        print("Processing in batches is not supported for this task.")
        sys.exit(1)
        
    # Load input data (X) from a pickle file and labels (y) from a CSV file
    with open(f'{data_folder}/{modal}.pkl', 'rb') as f:
        tensor_list = pickle.load(f)
        
        # Check if elements are NumPy arrays, no need to call detach() for NumPy arrays
        if isinstance(tensor_list[0], np.ndarray):
            data = tensor_list
        else:
            data = [tensor.detach().numpy() for tensor in tensor_list]  # In case they are PyTorch tensors
        
        # if modal == "graph":
        #     X = data
        # else:
        X = np.array(data)

    df = pd.read_csv(f'{data_folder}/label.csv')
    idx = df['id'] != '1AA6'
    df = df[idx] # Remove the row with the non-standard amino acid
    # X = X[idx]
    y = df['label']
    fold_ids = df['fold_id']
    
    return X, y, fold_ids

def train(modal, batches):
    data_folder = './data/ProtDD'
    # Load data and perform 10-fold cross-validation
    X, y, fold_ids = load_data(modal, data_folder, batches)
    cv = GroupKFold(n_splits=10)
    
    fold_results = []

    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Split the data into training and testing sets for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize and train the XGBoost classifier
        model = XGBClassifier(
            learning_rate=0.05,
            n_estimators=2000,
            max_depth=7,
            seed=42,  # Use seed instead of random_state
            tree_method='hist',
            objective="binary:logistic",
            early_stopping_rounds=10
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
        )

        if not os.path.exists(f'{data_folder}/{modal}_model/'):
            os.makedirs(f'{data_folder}/{modal}_model/')
        # Save the trained model for each fold
        model.save_model(f"{data_folder}/{modal}_model/fold{i+1}.json")
        
        score = model.score(X_test, y_test)  # Replace with appropriate metric
        fold_results.append({'fold': i + 1, 'score': score})
        
    print(f"Mean {modal} Accuracy: {np.mean([result['score'] for result in fold_results]):.4f}")

def test(modal):
    data_folder = './data/ProtDD'
    # Load data and perform 10-fold cross-validation for testing
    batches = False
    X, y, fold_ids = load_data(modal, data_folder, batches)
    cv = GroupKFold(n_splits=10)

    fold_scores = []
    for i, (train_index, test_index) in enumerate(cv.split(X, y, groups=fold_ids)):
        # Load the trained model for each fold
        X_test, y_test = X[test_index], y.iloc[test_index]
        model = XGBClassifier()
        model.load_model(f"{data_folder}/{modal}_model/fold{i+1}.json")

        # Make predictions and calculate accuracy for each fold
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_scores.append(accuracy)

    # Print the mean accuracy across all folds
    print("Mean Accuracy:", np.mean(fold_scores))


# Outside method to process label.csv
# List to hold indices of rows to remove
# rows_to_remove = []

# vgae_model, pae_model, esm_model, concrete_model = load_models()
# script_dir = os.path.dirname(os.path.abspath(__file__))
# target_dir = os.path.join(script_dir, "..", "data", "ProtDD")

# # Normalize the path (resolve "..")
# data_folder = os.path.abspath(target_dir)

# # Attempt to read label.csv
# try:
#     df = pd.read_csv(f'{data_folder}/label.csv')
# # Create label.csv
# except:
#     with open(f'{data_folder}/amino_enzymes.txt', 'r') as f:
#         enzyme_ids = [line.strip() for line in f]

#     with open(f'{data_folder}/amino_no_enzymes.txt', 'r') as f:
#         non_enzyme_ids = [line.strip() for line in f]

#     fold_dataframes = []
#     for fold_id in range(10):
#         with open(f'{data_folder}/amino_fold_{fold_id}.txt', 'r') as f:
#             protein_ids = [line.strip() for line in f]

#         labels = [1 if id in enzyme_ids else 0 for id in protein_ids]
#         fold_df = pd.DataFrame({'id': protein_ids, 'label': labels, 'fold_id': fold_id})
#         fold_dataframes.append(fold_df)

#     df = pd.concat(fold_dataframes, ignore_index=True)
#     df.to_csv(f'{data_folder}/label.csv', index=False)

# print("Number of samples:", len(df))

# # Iterate through the HDF5 files and extract multimodal representations
# for i, hdf5_file in tqdm(enumerate(df['id']), total=len(df['id'])):
#     hdf5_path = f'/{data_folder}/data/{hdf5_file}.hdf5'
    
#     # Get multimodal representations from the HDF5 file using pre-trained models
#     multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_modalities(hdf5_path, esm_model, vgae_model, pae_model, concrete_model)
    
#     if any(x is None for x in [multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud]):
#         rows_to_remove.append(i)
            
# # Remove the rows with non-standard amino acids
# df_cleaned = df.drop(rows_to_remove)

# print("Ending number of samples:", len(df_cleaned))

# # Save the cleaned label CSV
# df_cleaned.to_csv(f'{data_folder}/label_cleaned.csv', index=False)
# print(f"Number of samples after cleaning: {len(df_cleaned)}")
