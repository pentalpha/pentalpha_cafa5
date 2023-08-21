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
import glob

from go_classifier import GoClassifier

from prepare_data import *
from node import (Node, create_node, get_subnodes, node_factory, node_from_json_untrained, 
                  node_tsv, remove_redundant_nodes, 
                  node_tsv_results, load_node_depths, node_from_json)
from post_processing import solve_probabilities
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
if not path.exists(configs['whole_model_path']):
    mkdir(configs['whole_model_path'])
nodes_df_path = path.join(configs['whole_model_path'], 'classification_nodes.tsv')
nodes_tree_path = path.join(configs['whole_model_path'], 'classification_tree.tsv')

def select_parent_nodes():
    print('select_parent_nodes')
    train_terms_all = load_train_terms()

    classes = {}
    to_keep = []
    for goid, goid_group in train_terms_all.groupby("term"):
        if len(goid_group) >= 40:
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
datasets_dir = path.join(configs['whole_model_path'], configs['datasets_dir'])
models_dir = path.join(configs['whole_model_path'], configs['models_dir'])
processes = configs['processes']
test_size = configs['test_size']
experiments_dir = "experiments"
if not path.exists(experiments_dir):
    mkdir(experiments_dir)

def train_node(node_path, train_terms_all, train_protein_ids, train_plm_embeddings):
    
    node = node_from_json_untrained(node_path)
    node.set_model_path(models_dir)
    '''if path.exists(node.model_path):
        return True'''
    node.create_train_dataset(datasets_dir, train_terms_all, 
                                train_protein_ids, train_plm_embeddings,
                                test_size)
    node.train()
    #node.test_rock_auc_error()
    os.remove(node_path)
    if not node.failed:
        #pickle_file_path = path.join(datasets_dir, node.name.lstrip('GO:') + '_node.obj')
        #node.erase_dataset()
        node.to_json()
        return True
    else:
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
        #successes = [True]

    if successes:
        go_node_paths = [go_node_params[i][0] for i in range(len(successes)) if successes[i]]
        go_nodes_loaded = [node_from_json(go_node_path) for go_node_path in go_node_paths]

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
        node.to_json_untrained()

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

    
    for results_path, aspect in results_paths:
        print('Starting up classiifer')
        go_classifier = GoClassifier(nodes_df_path, 
                                     models_dir, datasets_dir, 
                                     load_go_graph(), aspect)
        go_classifier.set_limit()

        print('Loading terms, ids, and embeddings')
        test_protein_ids, test_plm_embeddings = load_test(
            aspect, configs['deepfried_path'])

        '''test_set_names = test_protein_ids[:5000]
        test_set_features = test_plm_embeddings[:5000]'''

        

        print('Splitting test')
        n_views_objective = int(len(test_protein_ids)/7000)
        id_views = np.array_split(test_protein_ids, n_views_objective)
        feature_views = np.array_split(test_plm_embeddings, n_views_objective)
        n_views = len(id_views)
        print(n_views, 'views created')
        print('Running predictions')
        outputs = []
        view_index = 0
        views_bar = tqdm(total=n_views)
        for features, ids in zip(feature_views, id_views):
            output_path = path.join(test_results_dir, 
                aspect+'_probs_'+str(view_index)+'.tsv')
            go_classifier.classify_proteins(features, ids, output_path)
            outputs.append(output_path)
            view_index += 1
            views_bar.update(1)
        views_bar.close()
        '''n_classifications = go_classifier.create_classifications(test_set_features,
                                                                test_set_names,
                                                                results_path)
        
        print('Created', n_classifications, 'classifications')'''

############################################################################

#nodes_df_path = 'classification_nodes.tsv'
#nodes_tree_path = 'classification_tree.tsv'

final_result = path.join(test_results_dir, 'final_ia.tsv')

def create_ia_probs():
    '''go_graph = load_graph()
    node_depths = load_node_depths(nodes_df_path)

    ia_files_basepaths = [test_results_dir + '/mf_predictions_*.tsv',
                          test_results_dir + '/cc_predictions_*.tsv',
                          test_results_dir + '/bp_predictions_*.tsv']
    ia_files = (glob.glob(ia_files_basepaths[0])
                +glob.glob(ia_files_basepaths[1])
                +glob.glob(ia_files_basepaths[2]))
    print('Found', len(ia_files), 'ia_files')
    #sorting by node depth
    ia_files.sort(key=lambda p: int(p.rstrip('.tsv').split('_')[-1]))
    probs = []
    for ia_file in tqdm(ia_files):
        print('Processing', ia_file)
        current_depth = int(ia_file.rstrip('.tsv').split('_')[-1])
        solve_probabilities(ia_file, go_graph, node_depths, probs, current_depth)
    
    final_ia = open(final_result, 'w')
    for prob_line in probs:
        newline = prob_line[0]+'\t'+prob_line[1]+'\t'+str(prob_line[2]) + '\n'
        final_ia.write(newline)
    final_ia.close()'''
    pass
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