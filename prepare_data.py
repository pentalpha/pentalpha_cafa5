from os import path, mkdir
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import subprocess
import os
import pcmap
import multiprocessing
import pickle
import networkx as nx
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

def load_train():
    train_terms = pd.read_csv(train_terms_path, sep="\t")
    train_protein_ids = np.load(train_protein_ids_path)
    train_plm_embeddings = np.load(train_plm_embeddings_path)

    # Now lets convert embeddings numpy array(train_embeddings) into pandas dataframe.
    column_num = train_plm_embeddings.shape[1]
    return train_terms, train_protein_ids, train_plm_embeddings, None

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

def download_pdbs():
    print('Loading protein ids')
    train_protein_ids = np.load(train_protein_ids_path)
    print(train_protein_ids[0])
    af_ids = []

    for x in train_protein_ids:
        af = 'AF-'+x+'-F1-model_v4.pdb'
        af_ids.append(af)

    for file in tqdm(af_ids):
        savepath = alphafold_dir + '/' + file
        tmp_path = alphafold_dir + '/' + file + '.tmp'
        if path.exists(tmp_path):
            run_command(['rm', tmp_path])
        if not path.exists(savepath):
            url = alphafold_url + file
            run_command(['wget --quiet', url, '-O', tmp_path, 
                         '&&', 'mv', tmp_path, savepath])
def entry_to_str(entry):
    return entry['resID'].rstrip().lstrip()+'_'+ entry['chainID']

def cc_json_to_matrix(pdb_path):
    cmap_path = pdb_path.replace('.pdb', '.cmap')
    if path.exists(cmap_path):
        return True
    cc_json = {}
    try:
        cc_json = pcmap.contactMap(pdb_path, dist=8.0)
    except TypeError as err:
        print(err)
        print('Could not parse', pdb_path)
        return False
    atoms = set()
    partner_list = []
    for entry in cc_json['data']:
        a = entry_to_str(entry['root'])
        atoms.add(a)
        partners = [entry_to_str(part) for part in entry['partners']]
        partner_list.append((a, partners)) 
        atoms.update(partners)
    atoms = list(atoms)
    #atoms.sort()
    index_dict = {atoms[i]: i for i in range(len(atoms))}
    atom_partner_lists = [np.zeros(len(atoms), dtype=np.float32) 
                              for i in range(len(atoms))]
    for a, partners in partner_list:
        x = index_dict[a]
        for b in partners:
            y = index_dict[b]
            atom_partner_lists[x][y] = 1.0
            atom_partner_lists[y][x] = 1.0
    
    contact_map = np.asarray(atom_partner_lists)
    np.save(open(cmap_path, 'wb'), contact_map)
    return True

'''
    #partnership_list = []
    #chains = set()    for entry in cc_json['data']:
        chains.add(entry['root']['chainID'])
        main = entry['root']['resID'].rstrip().lstrip()+'_'+ entry['root']['chainID']
        atoms.add(main)
        #print(entry['partners'])
        partners = [part['resID'].rstrip().lstrip()+'_'+ part['chainID'] 
                    for part in entry['partners']]
        atoms.update(partners)
        for partner in partners:
            partnership_list.append((main, partner))
    if len(chains) > 1:
        print(len(chains), 'in', pdb_path)
        return False
    else:
        atom_list = [int(x.split('_')[0]) for x in atoms]
        partnership_list_int = [(int(a.split('_')[0]), int(b.split('_')[0])) for a,b in partnership_list]
        atom_list.sort()
        atom_partner_lists = [np.zeros(len(atom_list), dtype=np.float32) 
                              for atom in atom_list]
        for a, b in partnership_list_int:
            atom_partner_lists[a-1][b-1] = 1.0
        contact_map = np.asarray(atom_partner_lists)
        np.save(open(pdb_path.replace('.pdb', '.cmap'), 'wb'), contact_map)
        return True'''

cmaps_path = 'alphafold/cmaps.npy'
cmaps_uniprotids = 'alphafold/cmaps_ids.obj'
def create_contact_maps():
    print('Loading data')
    _, train_protein_ids, train_plm_embeddings, train_pdbs = load_train()
    train_pdbs = load_pdbs(train_protein_ids)
    
    train_pdbs = [train_pdbs[i] for i in range(len(train_pdbs))
                if train_pdbs[i]]
    with multiprocessing.Pool(6) as pool:
        print('Calculating contact maps')
        bar = tqdm(total=len(train_pdbs))
        for pdb_chunk in chunks(train_pdbs, 2000):
            for result in pool.map(cc_json_to_matrix, pdb_chunk):
                bar.update(1)
            #print('Finished chunk of length', len(pdb_chunk))
        bar.close()

    print('Finding usable cmaps')
    train_cmaps = [pdb_path.replace('.pdb', '.cmap') for pdb_path in train_pdbs]
    train_cmaps = [cmap_path for cmap_path in train_cmaps if path.exists(cmap_path)]
    cmap_uniprotids = [cmap_path.split('-')[1] for cmap_path in train_cmaps]

    print('Loading them')
    cmaps = []
    for cmap_path in tqdm(train_cmaps):
        cmaps.append(np.load(cmap_path))

    cmaps = np.asarray(cmaps, dtype=np.int32)
    np.save(open('alphafold/cmaps.npy', 'wb'), cmaps)
    pickle.dump(cmap_uniprotids, open(cmaps_uniprotids, 'wb'))
    '''print('Filtering')
    train_protein_ids = np.array([train_protein_ids[i] for i in range(len(train_protein_ids))
                         if train_pdbs[i]])
    train_plm_embeddings = np.array([train_plm_embeddings[i] for i in range(len(train_plm_embeddings))
                         if train_pdbs[i]])
    
    np.save(open(train_plm_embeddings_with_prot_path, 'wb'), train_plm_embeddings)
    np.save(open(train_protein_ids_with_prot_path, 'wb'), train_protein_ids)'''
    #for train_pdb in tqdm(train_pdbs):
    #    success = cc_json_to_matrix(train_pdb)
        
    
if __name__ == "__main__":
    download_pdbs()
    #create_contact_maps()