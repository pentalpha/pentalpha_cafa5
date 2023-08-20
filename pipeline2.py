from multiprocessing import Pool
import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np
from os import path, mkdir
from tqdm import tqdm
import pickle
import sys
import json
import os
from datetime import datetime
import networkx as nx

from go_classifier import GoClassifier

from prepare_data import *
from node import (Node, create_node, get_subnodes, node_factory, 
                  node_tsv, remove_redundant_nodes, 
                  node_tsv_results, load_node_depths)
#base_dir_datasets = ''
#base_dir_datasets = '/kaggle/input/d/pentalpha'

def embedding_mean(emblist):
    total_emb = [sum(emblist[j][i] for j in range(len(emblist)))
                for i in range(len(emblist[0]))]
    return np.array([x/len(emblist) for x in total_emb])

def sep_validation(np_array, val_perc=0.15):
    val_start = int(len(np_array) * (1 - val_perc))
    print(val_start)
    train_x, val_x = np.split(np_array, [val_start])
    return train_x, val_x

configs = json.load(open("config.json", 'r'))
if not path.exists('tmp'):
    mkdir('tmp')

three_aspects = ['mf', 'cc', 'bp']
############################################################################

#Outputs
nodes_df_path = 'classification_nodes.tsv'
nodes_tree_path = 'classification_tree.tsv'

def select_parent_nodes():
    print('select_parent_nodes')
    train_terms_all = load_train_terms()

    classes = {}
    to_keep = []
    for goid, goid_group in train_terms_all.groupby("term"):
        if len(goid_group) >= 30:
            classes[goid] = len(goid_group)
            to_keep.append(goid)
            
    allowed_classes = list(classes.keys())
    train_terms_all = train_terms_all[train_terms_all['term'].isin(allowed_classes)]
    print(len(classes), 'classes to classify')
    print(len(train_terms_all), 'annotations')
    print(len(train_terms_all['EntryID'].unique().tolist()), 'proteins')

    graph = load_go_graph()
    removed = 0
    for goid in list(graph.nodes):
        if not str(goid) in to_keep:
            graph.remove_node(goid)
            removed += 1
    
    print('Removed', removed, 'nodes from gene ontology graph')
    #All GO
    #roots = [('GO:0003674', 'mf'), ('GO:0005575','cc'), ('GO:0008150', 'bp')]
    #Just MF
    #roots = [('GO:0003674', 'mf')]
    #Testsubset of MF
    #roots = [('GO:0003824', 'mf'), ('GO:0005488','mf'), ('GO:0005215', 'mf')]
    roots = configs['roots']

    id_to_name = {id_: data.get('name') 
        for id_, data in graph.nodes(data=True)}
    id_parents = [(id_, len(list(graph.successors(id_))))
        for id_, data in graph.nodes(data=True)]
    id_parents.sort(key=lambda x: x[1])

    root_nodes = []
    edges = []
    for root, aspect in roots:
        new_nodes, new_edges = create_node(root, classes, graph, id_to_name, aspect, 
                                max_children=configs['max_children'])
        root_nodes += new_nodes
        edges += new_edges
    
    node_list1 = get_subnodes(root_nodes)
    print(len(node_list1), 'nodes')
    node_list2 = remove_redundant_nodes(node_list1)
    print(len(node_list2), 'nodes')
    node_df = node_tsv(node_list2)
    open(nodes_df_path, 'w').write(node_df)
    tree_str = '\n'.join([x+'\t'+y for x, y in edges])
    open(nodes_tree_path, 'w').write(tree_str)

############################################################################
#inputs
#nodes_df_path

#Outputs
datasets_dir = configs['datasets_dir']
models_dir = configs['models_dir']
processes = configs['processes']
test_size = configs['test_size']
experiments_dir = "experiments"
if not path.exists(experiments_dir):
    mkdir(experiments_dir)

def train_node(node_path, train_terms_all, train_protein_ids, train_plm_embeddings):
    no_success = True
    tries = 3
    while no_success > 0:
        node = pickle.load(open(node_path, 'rb'))
        node.create_train_dataset(datasets_dir, train_terms_all, 
                                  train_protein_ids, train_plm_embeddings)
        node.train(models_dir, test_size=test_size)
        if not node.failed:
            pickle_file_path = path.join(datasets_dir, node.name.lstrip('GO:') + '_node.obj')
            node.erase_dataset()
            pickle.dump(node, open(pickle_file_path, 'wb'))
            return True
        else:
            tries -= 1

    os.remove(node_path)
    return False

def train_aspect(go_nodes, aspect):
    print('Training', aspect, 'with', len(go_nodes), 'nodes')
    print('Loading terms, ids, and embeddings')
    train_terms_all, train_protein_ids, train_plm_embeddings, _ = load_train(
        aspect, configs['deepfried_path'])

    experiment_name = "GO_Clf_Training_" + aspect
    experiment_path = path.join(experiments_dir, experiment_name)
    '''try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        mlflow.create_experiment(experiment_name)
    mlflow.start_run()'''

    if not path.exists(experiment_path):
        mkdir(experiment_path)
    
    start_run = datetime.now()
    day, hour = start_run.isoformat().replace('-','_').split('T')
    run_path = path.join(experiment_path, 
        day + '-' + hour.replace(':','_').split('.')[0])
    mkdir(run_path)
    experiment = {'start': datetime.now().isoformat()}
    #go_nodes = go_nodes[-2:]
    experiment["GO Nodes"] = len(go_nodes)
    experiment["train_plm_embeddings"] = len(train_plm_embeddings)
    experiment["train_terms_all"] = len(train_terms_all)

    '''mlflow.log_param("GO Nodes", len(go_nodes))
    mlflow.log_param("train_plm_embeddings", len(train_plm_embeddings))
    mlflow.log_param("train_terms_all", len(train_terms_all))'''

    go_node_params = [(node.path, train_terms_all, train_protein_ids, train_plm_embeddings)
                      for node in go_nodes]
    successes = None
    if processes > 1:
        with Pool(processes) as p:
            successes = p.starmap(train_node, go_node_params)
    else:
        successes = [train_node(*params) for params in go_node_params]

    if successes:
        go_node_paths = [go_node_params[i][0] for i in range(len(successes)) if successes[i]]
        go_nodes_loaded = [pickle.load(open(go_node_path, 'rb')) for go_node_path in go_node_paths]

        roc_auc_list = [node.roc_auc_score for node in go_nodes_loaded]
        evol_times_list = [node.evol_time for node in go_nodes_loaded]
        metaparams_list = [str(node.roc_auc_score)+': '+str(node.best_params) 
            for node in go_nodes_loaded]
        metaparams_txt = '\n'.join(metaparams_list) + '\n'

        nodes_df_txt = node_tsv_results(go_nodes_loaded)

        '''mlflow.log_metric("Min Evol. Time",  min(evol_times_list))
        mlflow.log_metric("Avg. Evol. Time", np.mean(evol_times_list))
        mlflow.log_metric("Max. Evol. Time", max(evol_times_list))

        mlflow.log_metric("Min ROC AUC",  min(roc_auc_list))
        mlflow.log_metric("Avg. ROC AUC", np.mean(roc_auc_list))
        mlflow.log_metric("Max. ROC AUC", max(roc_auc_list))'''

        '''open('tmp/metaparameters_list.txt', 'w').write(metaparams_txt)
        mlflow.log_artifact('tmp/metaparameters_list.txt')

        open('tmp/nodes_df.tsv', 'w').write(nodes_df_txt)
        mlflow.log_artifact('tmp/nodes_df.tsv')'''

        experiment["Min Evol. Time"] = min(evol_times_list)
        experiment["Avg. Evol. Time"] = np.mean(evol_times_list)
        experiment["Max. Evol. Time"] = max(evol_times_list)
        experiment["Min ROC AUC"] = min(roc_auc_list)
        experiment["Avg. ROC AUC"] = np.mean(roc_auc_list)
        experiment["Max. ROC AUC"] = max(roc_auc_list)
        end_run = datetime.now()
        experiment["end"] = end_run.isoformat()
        duration = ((end_run-start_run).seconds)/60
        experiment["duration"] = str(duration)+' minutes'
        open(run_path+'/metaparameters_list.txt', 'w').write(metaparams_txt)
        open(run_path+'/nodes_df.tsv', 'w').write(nodes_df_txt)
        experiment_str = json.dumps(experiment,indent=2)
        open(run_path+'/stats.json', 'w').write(experiment_str)

    mlflow.end_run() 

def create_node_datasets():

    go_nodes = node_factory(nodes_df_path)

    if not path.exists(datasets_dir):
        mkdir(datasets_dir)
    if not path.exists(models_dir):
        mkdir(models_dir)

    go_node_paths = [path.join(datasets_dir, node.goid.lstrip('GO:') + '_node.obj')
                        for node in go_nodes]
    for node in tqdm(go_nodes):
        node.set_path(datasets_dir)
        pickle.dump(node, open(node.path, 'wb'))

    node_by_aspect = {'mf': [], 'cc': [], 'bp': []}

    for go_node in go_nodes:
        node_by_aspect[go_node.aspect].append(go_node)

    for aspect, go_nodes_sublist in node_by_aspect.items():
        train_aspect(go_nodes_sublist, aspect)

############################################################################

#Outputs
test_results_dir = configs['results_dir']
results_paths = [(path.join(test_results_dir, x+'_predictions.tsv'),
                 x)
                 for x in three_aspects]

def load_graph():
    tree_str = open(nodes_tree_path, 'r').read().rstrip('\n').split("\n")
    edges = [(line.split('\t')[0],
              line.split('\t')[1]) 
              for line in tree_str]
    graph = nx.from_edgelist(edges)
    return graph

def classify_proteins():
    if not path.exists(test_results_dir):
        mkdir(test_results_dir)

    print('Loading terms, ids, and embeddings')
    for results_path, aspect in results_paths:
        test_protein_ids, test_plm_embeddings = load_test(
            aspect, configs['deepfried_path'])

        test_set_names = test_protein_ids[:1000]
        test_set_features = test_plm_embeddings[:1000]

        print('Test set has', len(test_set_names), 'proteins')

        go_classifier = GoClassifier(models_dir, datasets_dir, 
                                     load_go_graph(), aspect)

        n_classifications = go_classifier.create_classifications(test_set_features,
                                                                test_set_names,
                                                                results_path)
        
        print('Created', n_classifications, 'classifications')

############################################################################

#nodes_df_path = 'classification_nodes.tsv'
#nodes_tree_path = 'classification_tree.tsv'

final_result = path.join(test_results_dir, 'final_ia.tsv')
def choose_prob(prot_id, goid, preds, solved_probs, go_graph):
    '''print(prot_id, 'has', len(preds), 'predictions of', goid)
    #print(len(solved_probs), 'current solved of', prot_id)
    for parent_go, prob in preds:
        print('\t', parent_go, prob)'''
    parent_and_prob = [(parent_go.split('_')[0], prob) 
        for parent_go, prob in preds]
    parent_probs = {}
    for parent, _ in parent_and_prob:
        for _, _, goid, prob in solved_probs:
            if parent == goid:
                parent_probs[parent] = prob
                break
        if not parent in parent_probs:
            parent_probs[parent] = -1
    '''if -1 in parent_probs.values():
        print(parent_probs)
        for _, _, goid, prob in solved_probs:
            print(goid, prob)
            quit()'''
    parent_and_prob.sort(key = lambda x: parent_probs[x[0]])
    parent_higher, prob_higher = parent_and_prob[-1]
    
    return parent_higher, prob_higher

def solve_protein(solved_probs, protein_preds, go_graph, depth):
    prot_id = protein_preds[0][1]
    current_solved = [x for x in solved_probs if x[1] == prot_id]
    new_solved = []
    by_goid = {}
    for clf_name, prot_id, goid, prob in protein_preds:
        if not goid in by_goid:
            by_goid[goid] = []
        by_goid[goid].append([clf_name, prob])

    n_to_solve = 4
    for goid, preds in by_goid.items():
        if len(preds) == 1 or depth == 0:
            pred = preds[0]
            new_solved.append([pred[0], prot_id, goid, pred[1]])
        else:
            clf_name, prob = choose_prob(prot_id, goid, preds, 
                current_solved, go_graph)
            new_solved.append([clf_name, prot_id, goid, prob])
            #quit()

    return new_solved
        
def solve_probabilities(ia_file, go_graph, node_depths):
    depths = sorted(list(set(node_depths.values())))
    predictions_by_depth = {d: [] for d in depths}
    probs_stream = open(ia_file, 'r')
    header = probs_stream.readline()

    print('Reading probabilities')
    for rawline in probs_stream.readlines():
        clf_name, prot_id, goid, prob_str = rawline.rstrip('\n').split('\t')
        depth = node_depths[clf_name]
        predictions_by_depth[depth].append(
            [clf_name, prot_id, goid, float(prob_str)])
    
    solved_probs = []

    for depth, preds in predictions_by_depth.items():
        print(len(preds), 'predictions in depth', depth)
        protein_ids = set()
        for clf_name, prot_id, goid, prob in preds:
            protein_ids.add(prot_id)
        
        for protein_id in protein_ids:
            protein_preds = [x for x in preds if x[1] == protein_id]
            new_solved = solve_protein(solved_probs, protein_preds, go_graph, depth)
            solved_probs += new_solved
        
        print(len(solved_probs), 'solved predictions after depth', depth)
    return solved_probs

def create_ia_probs():
    go_graph = load_graph()
    node_depths = load_node_depths(nodes_df_path)

    ia_files = ['results_test/IA.tsv']
    probs = []
    for ia_file in ia_files:
        new_solved_probs = solve_probabilities(ia_file, go_graph, node_depths)
        probs += new_solved_probs
    
    final_ia = open(final_result, 'w')
    for prob_line in probs:
        newline = prob_line[1]+'\t'+prob_line[2]+'\t'+str(prob_line[3]) + '\n'
        final_ia.write(newline)
    final_ia.close()
############################################################################


if __name__ == "__main__":

    start_at = 0
    stop_at = 999
    if len(sys.argv) >= 2:
        start_at = int(sys.argv[1])
    if len(sys.argv) >= 3:
        stop_at = int(sys.argv[2])
    
    steps = [
        select_parent_nodes,
        create_node_datasets,
        classify_proteins,
        create_ia_probs
    ]

    to_run = steps[start_at:] if start_at > 0 else steps
    print(len(to_run), 'steps to run')
    i = start_at
    for step in to_run:
        print(i)
        step()
        if i == stop_at:
            break
        i += 1