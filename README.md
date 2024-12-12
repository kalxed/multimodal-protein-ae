# CS541 Final Group Project 

by Alexsandra Antoski, Kai Davidson, and Peter Howell

## Setup

We use conda to manage our project environment.

```
git clone https://github.com/HySonLab/Protein_Pretrain.git
cd multimodal-protein-ae
```
If you don't already have the conda environment:
```
conda env create -f environment.yml
```
Then run regardless:
```
conda activate dlprotproj
```
Finally if you already have this environment created update it with the following, ensuring you have the correct environment active:
```
conda env update --file environment.yml --prune
```

If you add new packages use this command to update the `environment.yml` file:

```conda env export | grep -v "^prefix: " > environment.yml```

## Folder Structure
Our project has the following structure
```
multimodal-protein-ae
├── README.md
├── data
│   ├── graphs/*.pt
│   ├── pointclouds/*.pt
│   ├── raw-structures/*.[cif | pdb]
│   └── sequences/*.pt
├── models
│   ├── CAE.pt
│   ├── PAE.pt
│   └── VGAE.pt
├── paper
├── mpae
│   ├── __init__.py
│   ├── mpae.py
│   ├── nn
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── concrete_autoencoder.py
│   │   ├── esm.py
│   │   ├── pae.py
│   │   └── vgae.py
│   └── utils
│       ├── __init__.py
│       ├── data.py
│       └── utils.py
└── scripts
    ├── py
    │   ├── fuse_encode.py
    │   ├── get_protein_list.py
    │   ├── make_graphs.py
    │   ├── make_pointclouds.py
    │   ├── make_tokenized_seqs.py
    │   ├── pretrain_pae.py
    │   └── pretrain_vgae.py
    └── sh
        ├── construct-graphs.sh
        ├── download-data.sh
        ├── get-protein-ids.sh
        ├── get-protein-structures.sh
        ├── graph-pretrain.sh
        ├── graph-test.sh
        ├── make-fusion.sh
        ├── make-pointclouds.sh
        └── pointcloud-pretrain.sh
```
### mpae
The `mpae` directory contains the classes and functions essential for our model. 

### scripts
`scripts` has the scripts used to deploy the model. 
#### scripts/py
contains the python scripts that were used to perform the data transformation, train the models, and evaluate them.
#### scripts/sh
contains the bash scripts used to submit jobs on the slurm cluster and download the data.

### data
this directory is where all the data was stored. The raw structure files downloaded from PDB were all placed in `data/raw-structures`. A graph, tokenized sequence, and point cloud was created from each of raw structure file and stored the respective directories.

### models
contains the pretrained models.
