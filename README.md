
## Requisites

kaggle API (for downloading datasets) and Python 3.10

## Preparing basic CAFA5 datasets

```
$ mkdir cafa-5-protein-function-prediction
$ mkdir t5embeds
$ cd cafa-5-protein-function-prediction && kaggle competitions download -c cafa-5-protein-function-prediction
$ cd ../t5embeds && kaggle datasets download -d sergeifironov/t5embeds
```

## Preparing structural features from DeepFRI

```
$ mkdir deepfried && cd deepfried && kaggle datasets download -d pentalpha/deepfriedproteins
```

This is the input data for the classification, containing sequence and structural features. It is created by a fork of the DeepFRI software, that I modified to predict features from the alphafold predicted PDBs of proteins included in the CAFA5 training and test sets. More information can be found here: https://github.com/pentalpha/DeepFRI

It creates contains the following files:
- 2,6G train_features_bp.npy  
- 2,6G train_features_mf.npy  
- 2,6G train_features_cc.npy  
- 5,1M train_ids_cc.npy
- 5,1M train_ids_bp.npy       
- 5,1M train_ids_mf.npy
- 2,6G test_features_bp.npy  
- 2,6G test_features_mf.npy  
- 2,6G test_features_cc.npy  
- 5,1M test_ids_cc.npy
- 5,1M test_ids_bp.npy       
- 5,1M test_ids_mf.npy

The (train|test) features (mf|bp|cc).npy files contain N_SAMPLES x 2519 matrices, with features to train each protein. The IDs files contain the names of the proteins in the corresponding matrices. Features are calculated separatly for Molecular Function, Biological Process and Cellular Component because DeepFRI has separate models for each ontology.

## Install libraries

```
$ pip install -f requirements.txt
```

## Running the pipeline

```

$ python pipeline2.py

```

# Pipeline Steps

The goal of the CAFA5 challenge is to predict probabilities for gene ontology terms for around 140000 proteins. They provide a list of GO annotations for the train proteins and the project must submit a list of predictions for the test proteins.

## Step 1 - select parent nodes

First, this step filters the GO terms. Only terms assigned to at least 40 proteins are kept.

Then, it goes to each root of the gene ontology tree (by default, the three ontologies) and tries to prepare each to become a classification node. Both child terms and their descendants become predicted classes. However, if the number of classes exceeds a maximum, the algorithm separates the child term with the most descendants and prepares it to be a separate node (this is recursive). The process continues until the classification node has a number of classes equal or smaller than the maximum (currently 250). 

The result is a collection of nodes, each predicting a small number of gene ontology terms.

## Step 2 - create node classifiers

First, all the train features from DeepFRI are loaded into memory.

The nodes and their predicted classes (separated in the previous step) are individually processed here, with a DNN model being created for each one. The process below is repeated for each node.

A subset of the training data is created, containing only proteins annotated with the current predicted classes. If this subset is greater than 40000 proteins, it is reduced to this number.

A labels dataframe is created. Each row is a binary list where the Xth item is 1 when the Xth predicted class is annotated to the protein and 0 when it's not.

The features matrix and the labels dataframe are split into train_features, train_labels, test_features and test_labels using iterative_train_test_split, with the test set being 25% of the data.

A DNN with two hidden layers is created. It uses Adam optimizer, relu activation for the hidden layers and sigmoid activation for the final layer.

The model is evaluated using ROC AUC score. If the score is larger than 0.6, the model is saved.

## Step 3 - classify proteins

First, all the test features from DeepFRI are loaded into memory.

The test data is separated into chunks of 12000 proteins. The chunks are have their classes predicted by each trained classification node. 

There is some redundancy between the nodes, with some GO terms being predicted by more than one node. When a term A is predicted by the classifiers of terms B and C, the probability of B and C is compared and the one with the highest probability is chosen. This is done first for the second layer of classifiers, then for the third and so on...

This creates a series of dataframes where the rows contain the name protein, the predicted GO term and it's probability.

## Step 4 - post processing

One more problem remains: sometimes, a GO term may have a high probability, but it's parent GO term has a small probability. This contradicts the rules of GO, at which an annotation to a child term implies an annotation of the parent term. So, when this happens, the probability of the child term is reduced 20% for each antecedent with low probability.