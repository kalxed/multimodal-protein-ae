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

```
multimodal-protein-ae
├── README.md
├── data
│   ├── graphs
│   ├── sequences
│   └── structures
├── paper
└── scraping
```