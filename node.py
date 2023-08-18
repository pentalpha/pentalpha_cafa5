

from typing import List
from networkx import MultiDiGraph
import networkx as nx
from os import path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
from skmultilearn.model_selection import iterative_train_test_split
from sklearn import metrics
from time import time
import json
import random

random.seed(1337)

from metaheuristic_clf import ClassificationOptimizer
from prepare_data import chunks

configs = json.load(open("config.json", 'r'))

class Node():

    tsv_columns = ['name', 'goid', 'node_depth', 'child_nodes', 'classes']
    tsv_columns2 = ['name', 'goid', 'node_depth', 'n_classes', 'roc_auc_score', 'evol_time']

    def __init__(self, goid, subclasses, subnodes, depth, aspect, 
                 additional_name=None, custom_name=None) -> None:
        self.goid = goid
        self.subclasses = subclasses
        self.subnodes = subnodes
        self.depth = depth
        self.aspect = aspect

        if custom_name:
            self.name = custom_name
        elif additional_name:
            self.name = self.goid + '_' + additional_name
        else:
            self.name = self.goid

    def to_tsv(self):
        return (self.aspect
                + '\t' + self.name
                + '\t' + self.goid 
                + '\t' + str(self.depth) 
                + '\t' + ','.join([x.goid for x in self.subnodes]) 
                + '\t' + ','.join(self.subclasses))

    def to_tsv_result(self):
        return (self.aspect
                + '\t' + self.name
                + '\t' + self.goid 
                + '\t' + str(self.depth)
                + '\t' + str(len(self.subclasses))
                + '\t' + str(self.roc_auc_score)
                + '\t' + str(self.evol_time))

    def create_train_dataset(self, datasets_dir, train_terms_all, 
                             train_protein_ids_all, train_plm_embeddings, 
                             max_samples=30000):
        
        valid_terms = train_terms_all[train_terms_all['EntryID'].isin(train_protein_ids_all)]
        train_terms_updated = valid_terms[valid_terms['term'].isin(self.subclasses)]
        train_protein_ids = train_terms_updated['EntryID'].unique().tolist()

        #Add 5% of falses to dataset
        false_protein_ids = [prot for prot in train_protein_ids_all 
                             if not prot in train_protein_ids]
        max_falses = len(train_protein_ids)*0.05
        falses = random.sample(false_protein_ids, int(max_falses))
        train_protein_ids = train_protein_ids + falses

        train_protein_ids.sort()
        labels_file_path = path.join(datasets_dir, self.goid.lstrip('GO:') + '_labels.npy')
        features_file_path = path.join(datasets_dir, self.goid.lstrip('GO:') + '_features.npy')

        protein_name_to_line     = {train_protein_ids[i]: i 
                                    for i in range(len(train_protein_ids))}
        protein_name_to_line_all = {train_protein_ids_all[i]: i 
                                    for i in range(len(train_protein_ids_all))}
        goid_to_col              = {self.subclasses[i]: i 
                                    for i in range(len(self.subclasses))}
        train_labels             = [np.array([0.0]*len(self.subclasses), dtype=np.float32) 
                                    for x in train_protein_ids]
        print(self.goid + ':', len(self.subclasses), len(train_protein_ids))
        last_prot  = None
        go_indexes = []
        for index, row in train_terms_updated.iterrows():
            prot = protein_name_to_line[row['EntryID']]
            goid = goid_to_col[row['term']]

            if prot != last_prot and last_prot:
                #go_indexes.sort()
                #for go_index in go_indexes:
                #    #train_labels[last_prot][go_index] = 1.0
                labels_list = np.array([1.0 if i in go_indexes else 0.0 for i in range(len(self.subclasses))])
                train_labels[last_prot] = labels_list
                go_indexes = []
                #print(last_prot, train_labels[last_prot][:20])
            go_indexes.append(goid)
            last_prot = prot
        
        for go_index in go_indexes:
            train_labels[last_prot][go_index] = 1.0
        train_labels[last_prot] = np.array([1.0 if i in go_indexes else 0.0 for i in range(len(self.subclasses))])
        #print(last_prot, train_labels[last_prot][:20])
        go_indexes = []
        
        '''if len(train_labels) > max_samples:
            train_labels = train_labels[:max_samples]'''
        train_labels = np.asarray(train_labels, dtype=np.float32)
        
        

        features = [train_plm_embeddings[protein_name_to_line_all[prot_name]] 
                    for prot_name in train_protein_ids]
        features = np.asarray(features, dtype=np.float32)

        if len(train_labels) > max_samples:
            print(self.goid, 'has too many samples:', len(train_labels))
            new_perc = max_samples / len(train_labels)
            _, _, new_features, new_labels = iterative_train_test_split(features, train_labels, test_size = new_perc)

            features = new_features
            train_labels = new_labels
            print(len(train_labels), len(features))
        
        '''f = open(labels_file_path, 'wb')
        np.save(f, train_labels)
        f.close()
        f2 = open(features_file_path, 'wb')
        np.save(f2, features)
        f2.close()'''
        self.features = features
        self.train_labels = train_labels

        #self.features_file_path = features_file_path'''
        #self.labels_file_path = labels_file_path
        #self.train_protein_ids = train_protein_ids

    def train(self, models_dir, test_size=0.25):
        print('Splitting')
        train_features, train_labels, test_features, test_labels = iterative_train_test_split(self.features, 
            self.train_labels, test_size = test_size)
        
        start_time = time()
        try:
            ga = ClassificationOptimizer(train_features, test_features, 
                                    train_labels, test_labels, 
                                    gens=configs['gens'], pop=configs['pop'], 
                                    parents=configs['parents'])
            ga.run_ga()
            self.failed = False
        except ValueError as err:
            print(err)
            self.evol_time = time() - start_time
            self.roc_auc_score = 0.0
            self.best_params = {}
            self.failed = True
            return None
        
        self.evol_time = time() - start_time
        print('GA terminando após', self.evol_time/60, 'minutos')
        ga_info = [ga.solutions_by_generation, ga.fitness_by_generation, 
                ga.evolved, ga.finish_type, self.evol_time]
        annot_model = ga.reg
        print("Best Params:", ga.best_params)
        print('\n'.join([str(fitvec) for fitvec in ga.fitness_by_generation]))
        print('evolved', ga.evolved)
        print('ga.finish_type', ga.finish_type)
        print('evol_time', self.evol_time)

        print(annot_model.summary())

        self.roc_auc_score = ga.score
        self.best_params = ga.best_params

        self.model_path = path.join(models_dir, self.name.lstrip('GO:') + '_classifier.keras')
        annot_model.save(self.model_path)
        
    def erase_dataset(self):
        self.features = None
        self.train_labels = None
        self.annot_model = None

    def load_model(self):
        self.annot_model = tf.keras.models.load_model(self.model_path)

    def set_path(self, datasets_dir):
        self.path = path.join(datasets_dir, self.name.lstrip('GO:') + '_node.obj')

    def classify(self, input_X, protein_names):
        y_pred = self.annot_model.predict(input_X)
        assert len(y_pred[0]) == len(self.subclasses)
        predicted = []
        for protein_index in range(len(y_pred)):
            probs_vec = y_pred[protein_index]
            prot_name = protein_names[protein_index]
            prob_tuples = [(self.name, prot_name, 
                            self.subclasses[i], probs_vec[i]) 
                           for i in range(len(probs_vec))]
            predicted += prob_tuples
        
        return predicted

def node_factory(node_description_df_path: str) -> List[Node]:
    stream = open(node_description_df_path, 'r')
    new_nodes = []
    header = stream.readline()
    for rawline in stream.readlines():
        cells = rawline.rstrip('\n').split('\t')
        aspect, nodename, goid, node_depth, child_nodes, classes = cells
        new_nodes.append(Node(
            goid, classes.split(','), [], int(node_depth), 
            aspect, custom_name=nodename
        ))
    return new_nodes

def count_subclasses(children, children_descendants):
    subclasses = set(children)
    #print(subclasses)
    #print([type(x) for x in children_descendants])
    for c in children_descendants:
        subclasses.update(c)
    return subclasses

def create_subnodes_for_children(children, classes, graph, id_to_name, aspect, ident, max_children):
    children_descendants = [list(nx.ancestors(graph, child_id)) for child_id in children]
    n_subclasses = len(count_subclasses(children, children_descendants))
    subnodes = []
    subedges = []
    while n_subclasses > max_children:
        print('\t'*ident + 'Too big', n_subclasses)
        largest_class_index = 0
        for class_index in range(len(children)):
            if len(children_descendants[class_index]) > len(children_descendants[largest_class_index]):
                largest_class_index = class_index
        new_subnodes, new_edges = create_node(children[largest_class_index], classes, 
                                    graph, id_to_name, 
                                   aspect, ident = ident+1, max_children=max_children)
        subedges += new_edges
        children_descendants[largest_class_index] = []
        subnodes += new_subnodes
        n_subclasses = len(count_subclasses(children, children_descendants))
        print('\t'*ident + 'Reduced to', n_subclasses)
    
    return children_descendants, subnodes, subedges

def create_node(go_id, classes: dict, graph: MultiDiGraph, 
                id_to_name: dict, aspect: str, ident = 0, max_children = 250):
    print('\t'*ident + 'Making node for', id_to_name[go_id])
    children = list(graph.predecessors(go_id))
    n_newnode = 0
    nodes_to_return = []
    new_edges = []
    for children_sublist in chunks(children, max_children):
        children_descendants, subnodes, subedges = create_subnodes_for_children(
            children_sublist, 
            classes, graph, id_to_name, aspect, ident, max_children)
        new_edges += subedges
        all_subclasses = list(count_subclasses(children, children_descendants))
        new_edges += [(go_id, subclass) for subclass in all_subclasses]
        print('\t'*ident + 'Made node for', id_to_name[go_id], 'with', 
            len(children), '+', [len(x) for x in children_descendants])
        name_sufix = None if n_newnode == 0 else str(n_newnode)
        new_node = Node(go_id, all_subclasses, subnodes, ident, aspect,
                additional_name = name_sufix)
        nodes_to_return.append(new_node)
        n_newnode += 1
    return nodes_to_return, new_edges

def get_subnodes(nodes: List[Node]):
    new_nodes = []
    for node in nodes:
        new_nodes.append(node)
        for subnode in node.subnodes:
            new_nodes.append(subnode)
            if len(subnode.subnodes) > 0:
                new_nodes += get_subnodes(subnode.subnodes)
    return new_nodes

def node_list(nodes: List[Node]):
    if len(nodes) > 1:
        new_nodelist = []
        for node in nodes:
            new_nodelist += node_list([node])
        return new_nodelist
    elif len(nodes) == 1:
        new_nodelist = [nodes[0]]
        for subnode in nodes[0].subnodes:
            new_nodelist.append(subnode)
        return new_nodelist
    elif len(nodes) == 0:
        return []
    
def remove_redundant_nodes(nodes: List[Node]):
    node_by_id = {}
    for node in nodes:
        node_id = node.goid+'_'+','.join(sorted(node.subclasses))
        if not node_id in node_by_id:
            node_by_id[node_id] = []
        node_by_id[node_id].append(node)
    
    node_list_final = []
    for parentid in node_by_id.keys():
        nodes = node_by_id[parentid]
        if len(nodes) > 1:
            print(parentid, 'has', len(nodes), 'nodes')
            nodes.sort(key=lambda n: n.depth)
        node_list_final.append(nodes[0])

    return node_list_final

def node_tsv(nodes: List[Node]):
    text = ['\t'.join(Node.tsv_columns)]
    nodes.sort(key = lambda node: (node.depth, -len(node.subnodes), -len(node.subclasses), node.goid))
    for node in nodes:
        text.append(node.to_tsv())
    return '\n'.join(text)

def node_tsv_results(nodes: List[Node]):
    text = ['\t'.join(Node.tsv_columns2)]
    nodes.sort(key = lambda node: (node.depth, -len(node.subclasses), node.goid))
    for node in nodes:
        text.append(node.to_tsv_result())
    return '\n'.join(text)