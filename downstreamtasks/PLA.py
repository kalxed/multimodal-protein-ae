import torch
import pickle
import pandas as pd
import sys
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from utils import *
from mpae.nn.esm import *
from mpae.nn.vgcn import *
from mpae.nn.pae import *
import gpytorch
import ast

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_ligand_representation(ligand_smiles):
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {ligand_smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
    fingerprint = generator.GetFingerprint(mol)
    fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float32)
    return fingerprint_tensor

def process(dataset_name, batches):
    """
    Process all modality with protein-ligand affinity data
    """
    # Load dataset
    data_folder = f'./data/{dataset_name}'
    df = pd.read_csv(f'{data_folder}/label.csv')
    print("Number of samples:", len(df))
    
    # Load pre-trained models
    vgae_model, pae_model, esm_model, concrete_model = load_models()

    mulmodal = []
    sequence = []
    graph = []
    point_cloud = []
    batch_size = 250  # Number of samples to process in each batch
    batch_counter = 0  # Keep track of batch number

    # Iterate through the dataset to process each sample
    for i, (ligand_smiles, protein_name) in tqdm(
        enumerate(zip(df["ligand"], df["protein"])),
        desc="Processing proteins and ligands",
        total=len(df),
        bar_format="{l_bar}{bar} | Elapsed: {elapsed}, Remaining: {remaining}",
    ):
        pdb_path = f"{data_folder}/pdb/{protein_name}.pdb"
        multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud = get_modalities(pdb_path, esm_model, vgae_model, pae_model, concrete_model)
        
        # Skip the sample if any of the modalities are None
        if any(x is None for x in [multimodal_representation, encoded_sequence, encoded_graph, encoded_point_cloud]):
            continue
        
        ligand_representation = get_ligand_representation(ligand_smiles)

        # Concatenate multimodal representation with ligand representation
        mulmodal_feature = torch.cat((multimodal_representation, ligand_representation), dim=0).detach().numpy()
        sequence_feature = torch.cat((encoded_sequence, ligand_representation), dim=0).detach().numpy()
        graph_feature = torch.cat((encoded_graph, ligand_representation), dim=0).detach().numpy()
        point_cloud_feature = torch.cat((encoded_point_cloud, ligand_representation), dim=0).detach().numpy()

        # Append the features to their respective lists
        mulmodal.append(mulmodal_feature)
        sequence.append(sequence_feature)
        graph.append(graph_feature)
        point_cloud.append(point_cloud_feature)
        
        # Save and reset data after reaching the batch size
        if batches:
            if (i + 1) % batch_size == 0 or (i + 1) == len(df):
                batch_counter += 1
                pickle_batch_dump(batch_counter, data_folder, mulmodal, sequence, graph, point_cloud)

                # Clear lists to free up memory
                mulmodal.clear()
                sequence.clear()
                graph.clear()
                point_cloud.clear()

    if not batches:
        pickle_dump(data_folder, mulmodal, sequence, graph, point_cloud)
    print("Features processed successfully.")

def setup(dataset, modal, batches):
    data_folder = f'./data/{dataset}'

    # Load feature data from the selected modality 
    if batches:
        if modal != "multimodal":
            modality_folder = os.path.join(data_folder, modal + "s")
        else:
            modality_folder = os.path.join(data_folder, modal)
        num_batches = len([name for name in os.listdir(modality_folder) if name.endswith('.pkl') and os.path.isfile(os.path.join(modality_folder, name))])
        print(f'Number of batches: {num_batches}')
        X = load_batch_data(modality_folder, modal)
    if not batches:
        # Check if the data is already processed
        if not os.path.exists(f'{data_folder}/{modal}.pkl'):
            print(f"Error: {modal} data not found. Please process the data first.")
            sys.exit(1)
        else: 
            X = load_data(data_folder, modal)

    # Read the label CSV file
    df = pd.read_csv(f'{data_folder}/label.csv')
    print("Number of samples:", len(df))
    
    y = df['label'].to_numpy()

    # Load test and train IDs from the fold setting
    with open(f'{data_folder}/folds/test_fold_setting.txt', 'r') as f:
        test_ids_str = f.read()
        test_ids = ast.literal_eval(test_ids_str)

        # Handle the case for graph data and non-graph data separately
        # if modal == "graph":
        #     train_ids = np.setdiff1d(np.arange(len(X)), test_ids)
        # else:
        train_ids = np.setdiff1d(np.arange(X.shape[0]), test_ids)

    # Print the sizes of the training and test sets
    print(f'Size of Training Set: {len(train_ids)} samples')
    print(f'Size of Test Set: {len(test_ids)} samples')
    
    # Ensure test_ids are within the valid range
    test_ids = [i for i in test_ids if i < len(X)]
    # print(f'Filtered Test IDs: {test_ids}')

    # Split the data into training and test sets
    # if modal == "graph":
    #     # If dealing with graphs, split the list of Data objects into train/test
    #     X_train = [X[i] for i in train_ids]
    #     X_test = [X[i] for i in test_ids]
        
    #     # Set the dimensionality of graphs based on the median length 
    #     lengths = [len(sample) for sample in X]
    #     median_length = int(np.median(lengths))
    #     X_train = [pad_truc(sample, median_length) for sample in X_train]
    #     X_test = [pad_truc(sample, median_length) for sample in X_train]
        
    # else:
    #     # For non-graph modalities (like sequence, point cloud, etc.)
    X_train = X[train_ids]
    X_test = X[test_ids]
        
    y_train = y[train_ids]
    y_test = y[test_ids]
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    return model, X_train, y_train, X_test, y_test, likelihood, data_folder

def train(dataset, modal, batches):
    # Define optimizer and loss
    model, X_train, y_train, X_test, y_test, likelihood, data_folder = setup(dataset, modal, batches)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.20)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Train the model
    training_iter = 100
    for i in range(training_iter):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        # Compute the output from the model
        output = model(X_train)
        # Calculate the loss and backpropagate gradients
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
    
    # Save the trained model state
    torch.save(model.state_dict(), f'{data_folder}/{modal}_model_state.pth')

def test(dataset, modal):
    model, X_train, y_train, X_test, y_test, likelihood, data_folder = setup(dataset)
    # Load the trained model state
    state_dict = torch.load(f'{data_folder}/{modal}_model_state.pth')
    model.load_state_dict(state_dict)

    # Evaluate the model on the test set
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test))
        mse = gpytorch.metrics.mean_squared_error(observed_pred, y_test, squared=True).item()
        mae = gpytorch.metrics.mean_absolute_error(observed_pred, y_test).item()
        ci = get_cindex(observed_pred.loc, y_test)
        rm2 = get_rm2(observed_pred.loc, y_test)
        pearsonr = get_pearson(observed_pred.loc, y_test)
        spearmanr = get_spearman(observed_pred.loc, y_test)

        lower, upper = observed_pred.confidence_region()

        # Print evaluation metrics
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("Root Mean Square Error: ", sqrt(mse))
        print("Pearson Correlation:", pearsonr)
        print("Spearman Correlation:", spearmanr)
        print("C-Index:", ci)
        print("Rm^2:", rm2)
        print("Mean of Lower Confidence Bounds:", lower.mean())
        print("Mean of Upper Confidence Bounds:", upper.mean())
        
        
# Outside method to process label.csv in PLA dataset
# dataset_name = "DAVIS"
# data_folder = f'./data/{dataset_name}'
# df = pd.read_csv(f'{data_folder}/label.csv')
# print("Starting number of samples:", len(df))

# # List to hold indices of rows to remove
# rows_to_remove = []

# # Iterate through the dataset to process each sample
# for i, (ligand_smiles, protein_name) in tqdm(
#     enumerate(zip(df["ligand"], df["protein"])),
#     desc="Evaluating proteins",
#     total=len(df),
#     bar_format="{l_bar}{bar} | Elapsed: {elapsed}, Remaining: {remaining}",
# ):
#     pdb_path = f"{data_folder}/pdb/{protein_name}.pdb"
    
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure('protein', pdb_path)

#     for residue in structure.get_residues():
#         if 'CA' in residue:
#             try:  
#                 aa_code = Polypeptide.three_to_index(residue.get_resname())
#                 has_non_standard_aa = False
#             except KeyError:
#                 # Non-standard residue, break and mark for removal
#                 has_non_standard_aa = True
#                 break
            
#     # If a non-standard amino acid is found, mark the row for removal
#     if has_non_standard_aa:
#         rows_to_remove.append(i)
            
# # Remove the rows with non-standard amino acids
# df_cleaned = df.drop(rows_to_remove)

# print("Ending number of samples:", len(df_cleaned))

# # Save the cleaned label CSV
# df_cleaned.to_csv(f'{data_folder}/label_cleaned.csv', index=False)
# print(f"Number of samples after cleaning: {len(df_cleaned)}")