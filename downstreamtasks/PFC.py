from utils import *
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def process(batches):
    if batches:
        print("Processing in batches is not supported for this task.")
        sys.exit(1)
    
    # Define the script directory and data folder
    vgae_model, pae_model, esm_model, cae_model = load_models()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "..", "data", "SCOPe1.75")
        
    hdf5_folder_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

    # Iterate through different folders containing text files
    for folder in hdf5_folder_name:
        with open(f'{data_folder}/{folder}.txt', 'r') as file:
            lines = file.readlines()
            hdf5_file_names = [line.split()[0] for line in lines]

        # Initialize lists to store multimodal representations
        mulmodal = []
        sequence = []
        graph = []
        point_cloud = []

        # Iterate through HDF5 files in the current folder
        for i, hdf5_file_name in tqdm(enumerate(hdf5_file_names), total=len(hdf5_file_names)):
            hdf5_path = f'{data_folder}/{folder}/{hdf5_file_name}.hdf5'

            # Get multimodal representations from the HDF5 file using pre-trained models
            multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_modalities(
                hdf5_path, esm_model, vgae_model, pae_model, cae_model)
            
            if any(x is None for x in [multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud]):
                continue

            # Append the representations to their respective lists
            mulmodal.append(multimodal_representation.detach().numpy())
            sequence.append(encoded_sequence.detach().numpy())
            graph.append(encoded_graph.detach().numpy())
            point_cloud.append(encoded_point_cloud.detach().numpy())

        # Save the multimodal representations to pickle files in the target folder
        with open(f'{data_folder}/{folder}/multimodal.pkl', 'wb') as f:
            pickle.dump(mulmodal, f)

        with open(f'{data_folder}/{folder}/sequence.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        with open(f'{data_folder}/{folder}/graph.pkl', 'wb') as f:
            pickle.dump(graph, f)

        with open(f'{data_folder}/{folder}/point_cloud.pkl', 'wb') as f:
            pickle.dump(point_cloud, f)

        print(f"Saved {folder} folder")
        
def setup(modal, mode, test_dataset):
    # Specify the data folder and the label file names
    data_folder = './data/SCOPe1.75'
    label_file_name = ['training', 'validation', 'test_family', 'test_fold', 'test_superfamily']

    # Initialize dictionaries to store feature data and labels
    X = {}
    y = {}

    # Load feature data and labels for each label file
    for file in label_file_name:
        # Load labels from the text file
        with open(f'{data_folder}/{file}.txt', 'r') as f:
            lines = f.readlines()
            y[file] = [line.split()[-1] for line in lines]

        # Load feature data from the selected modality
        with open(f'{data_folder}{file}/{modal}.pkl', 'rb') as f:
            X[file] = pickle.load(f)

    # Split data into train, validation, and test sets
    X_train, y_train = X["training"], y['training']
    X_validation, y_validation = X["validation"], y['validation']
    
    X_test1, y_test1 = X["test_family"], y['test_family']
    X_test2, y_test2 = X["test_fold"], y['test_fold']
    X_test3, y_test3 = X["test_superfamily"], y['test_superfamily']

    # Encode labels using LabelEncoder
    label = set(y_train)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(label))

    y_train = label_encoder.transform(y_train)
    y_validation = label_encoder.transform(y_validation)
    y_test1 = label_encoder.transform(y_test1)
    y_test2 = label_encoder.transform(y_test2)
    y_test3 = label_encoder.transform(y_test3)
    
    if mode == 'train':
        return modal, X_train, y_train, X_validation, y_validation
    elif mode == 'test':
        if test_dataset == 'test_family':
            return data_folder, modal, X_test1, y_test1
        elif test_dataset == 'test_fold':
            return data_folder, modal, X_test2, y_test2
        elif test_dataset == 'test_superfamily':
            return data_folder, modal, X_test3, y_test3
        else:
            print("Invalid test dataset. Use 'test_family', 'test_fold', or 'test_superfamily'.")

def train(modal, batches):
    data_folder = './data/SCOPe1.75'
    modal, X_train, y_train, X_validation, y_validation = setup(modal, 'train', None)
    
    # Train a multi-class classification model using XGBoost
    xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, random_state=42, tree_method='gpu_hist', objective='multi:softmax')
    xgb_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_train, y_train), (X_validation, y_validation)], early_stopping_rounds=10, verbose=1)
    
    # Save the trained model
    xgb_model.save_model(f"{data_folder}/{modal}_model.json")

def test(test_dataset, modal, batches):
    data_folder = './data/SCOPe1.75'
    modal, X_test, y_test = setup(modal, 'test', test_dataset)
    
    # Load the trained XGBoost classifier model
    xgb_model = XGBClassifier()
    xgb_model.load_model(f"{data_folder}/{modal}_model.json")

    # Make predictions and calculate accuracy for each test set
    y_pred = xgb_model.predict(X_test)
    print(f"Accuracy :", accuracy_score(y_test, y_pred))