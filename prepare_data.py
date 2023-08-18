from os import path, mkdir
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import subprocess
import obonet

train_terms_path = 'cafa-5-protein-function-prediction/Train/train_terms.tsv'
go_basic_obo_path = 'cafa-5-protein-function-prediction/Train/go-basic.obo'
train_protein_ids_path = 't5embeds/train_ids.npy'
train_protein_ids_with_prot_path = 't5embeds/train_ids_with_prot.npy'
train_plm_embeddings_path = 't5embeds/train_embeds.npy'
train_plm_embeddings_with_prot_path = 't5embeds/train_embeds_with_prot.npy'
alphafold_dir = 'alphafold'
protein2ipr_filtered_path = "interpro/protein2ipr_cafa5.dat"
alphafold_url = 'https://alphafold.ebi.ac.uk/files/'

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_command(cmd_vec: List[str], stdin="", no_output=True):
    '''Executa um comando no shell e retorna a saída (stdout) dele.'''
    cmd_vec = " ".join(cmd_vec)
    #logging.info(cmd_vec)
    if no_output:
        #print(cmd_vec)
        result = subprocess.run(cmd_vec, shell=True)
        return ""
    else:
        result = subprocess.run(cmd_vec, capture_output=True, 
            text=True, input=stdin, shell=True)
        return result.stdout

def find_alphafold(prot_ids):
    file_paths = [alphafold_dir+'/AF-'+x+'-F1-model_v4.pdb' for x in prot_ids]
    file_paths = [x if path.exists(x) else None for x in file_paths]
    return file_paths

def load_train_terms():
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    return train_terms

def load_train(aspect, deepfried_path):
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    deep_train_protein_ids_path = path.join(deepfried_path,
        'train_ids_'+aspect+'.npy')
    deep_train_plm_embeddings_path = path.join(deepfried_path,
        'train_features_'+aspect+'.npy')
    train_protein_ids = np.load(deep_train_protein_ids_path)
    train_plm_embeddings = np.load(deep_train_plm_embeddings_path)

    # Now lets convert embeddings numpy array(train_embeddings) into pandas dataframe.
    column_num = train_plm_embeddings.shape[1]
    return train_terms, train_protein_ids, train_plm_embeddings, None

def load_test(aspect, deepfried_path):
    deep_test_protein_ids_path = path.join(deepfried_path,
        'test_ids_'+aspect+'.npy')
    deep_test_plm_embeddings_path = path.join(deepfried_path,
        'test_features_'+aspect+'.npy')
    test_protein_ids = np.load(deep_test_protein_ids_path)
    test_plm_embeddings = np.load(deep_test_plm_embeddings_path)

    # Now lets convert embeddings numpy array(test_embeddings) into pandas dataframe.
    column_num = test_plm_embeddings.shape[1]
    return test_protein_ids, test_plm_embeddings

def load_go_graph():
    graph = obonet.read_obo(go_basic_obo_path)
    return graph

def load_pdbs(train_protein_ids):
    train_pdbs = find_alphafold(train_protein_ids)
    return train_pdbs

def load_categories_interpro():
    protein2ipr_filtered_path = "interpro/protein2ipr_cafa5.dat"
    category_input = open(protein2ipr_filtered_path, 'r')

    cat_of_proteins = {}
    used_categories = set()
    for rawline in category_input:
        cells = rawline.rstrip('\n').split('\t')
        if not cells[0] in cat_of_proteins:
            cat_of_proteins[cells[0]] = set()
        used_categories.add(cells[1])
        cat_of_proteins[cells[0]].add(cells[1])

    category_input.close()
    print(len(used_categories), 'used_categories')
    print(list(used_categories)[:10])

    return cat_of_proteins, used_categories

def entry_to_str(entry):
    return entry['resID'].rstrip().lstrip()+'_'+ entry['chainID']
