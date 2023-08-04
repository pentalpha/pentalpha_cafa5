from networkx import MultiDiGraph
#import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path, mkdir
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gzip
from lxml import etree
import re
from tqdm import tqdm
import pickle
import sys
from time import time
#from sklearn import metrics
#from sklearn.model_selection import train_test_split
#from metaheuristic import RegressionOptimizer
import networkx as nx

from prepare_data import *
from node import Node, create_node, get_subnodes, node_factory, node_list, node_tsv, remove_redundant_nodes
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

############################################################################


#Outputs
nodes_df_path = 'classification_nodes.tsv'

def select_parent_nodes():
    print('select_parent_nodes')
    train_terms_all, _, _, _ = load_train()

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
    roots = ['GO:0003674', 'GO:0005575', 'GO:0008150']

    id_to_name = {id_: data.get('name') 
        for id_, data in graph.nodes(data=True)}
    id_parents = [(id_, len(list(graph.successors(id_))))
        for id_, data in graph.nodes(data=True)]
    id_parents.sort(key=lambda x: x[1])

    root_nodes = []
    for root in roots:
        root_node = create_node(root, classes, graph, id_to_name)
        root_nodes.append(root_node)
    
    node_list1 = get_subnodes(root_nodes)
    print(len(node_list1), 'nodes')
    node_list2 = remove_redundant_nodes(node_list1)
    print(len(node_list2), 'nodes')
    node_df = node_tsv(node_list2)
    open(nodes_df_path, 'w').write(node_df)

############################################################################
#inputs
#nodes_df_path

#Outputs
datasets_dir = 'datasets'

def create_node_datasets():
    print('Loading terms, ids, and embeddings')
    train_terms_all, train_protein_ids, train_plm_embeddings, _ = load_train()

    go_nodes = node_factory(nodes_df_path)
    
    if not path.exists(datasets_dir):
        mkdir(datasets_dir)

    for node in tqdm(go_nodes):
        node.create_train_dataset(datasets_dir, train_terms_all, train_protein_ids, train_plm_embeddings)

        node.train()

        pickle_file_path = path.join(datasets_dir, node.goid.lstrip('GO:') + '_node.obj')
        node.erase_dataset()
        pickle.dump(node, open(pickle_file_path, 'wb'))

############################################################################
#Inputs
#labels_file_path
#final_annot_features_path
#train_plm_embeddings_path = 't5embeds/train_embeds.npy'

#Outputs:
multiclf_model_path = 'multiclf_model_a.obj'
multiclf_hist_path = 'multiclf_hist_a.obj'

def train_multclf_dnn(aspect_code, labels_file_path, train_plm_embeddings_train, train_plm_embeddings_val):
    train_labels = np.load(labels_file_path)

    print('Splitting')
    train_labels_train, train_labels_val = sep_validation(train_labels, val_perc=0.2)
    assert len(train_plm_embeddings_train) == len(train_labels_train)
    print('Creating model')
    INPUT_SHAPE = [train_plm_embeddings_train.shape[1]]
    print(train_plm_embeddings_train.shape)
    BATCH_SIZE = 5120

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),    
        tf.keras.layers.Dense(units=train_plm_embeddings_train.shape[1]*0.75, activation='relu'),
        tf.keras.layers.Dense(units=train_plm_embeddings_train.shape[1]*0.6, activation='relu'),
        tf.keras.layers.Dense(units=train_plm_embeddings_train.shape[1]*0.5, activation='relu'),
        tf.keras.layers.Dense(units=len(train_labels[0]),activation='sigmoid')
    ])


    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', tf.keras.metrics.AUC()],
    )

    history = model.fit(
        train_plm_embeddings_train, train_labels_train,
        validation_data=(train_plm_embeddings_val, train_labels_val),
        batch_size=BATCH_SIZE,
        epochs=6
    )

    pickle.dump(model, open(multiclf_model_path.replace('_a', aspect_code), 'wb'))
    pickle.dump(history, open(multiclf_hist_path.replace('_a', aspect_code), 'wb'))

def train_clf():
    print('train_clf')
    print('loading plm')
    _, _, train_plm_embeddings, _ = load_train()
    '''print('loading annot embedding')
    final_annot_features = np.load(final_annot_features_path)
    print('stacking')
    train_plm_embeddings = np.hstack((train_plm_embeddings, final_annot_features))'''
    print('load_labels')
    aspect_labels_paths = [('BP', labels_bp_file_path), 
                  ('MF', labels_mf_file_path), 
                  ('CC', labels_cc_file_path)]
    train_plm_embeddings_train, train_plm_embeddings_val = sep_validation(train_plm_embeddings, val_perc=0.2)
    for aspect_code, labels_file_path in aspect_labels_paths:
        print('Training for', aspect_code)
        train_multclf_dnn(aspect_code, labels_file_path, train_plm_embeddings_train, train_plm_embeddings_val)
        

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
        create_node_datasets
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