

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
from metaheuristic_clf import ClassificationOptimizer

class Node():

    tsv_columns = ['goid', 'node_depth', 'child_nodes', 'classes']
    tsv_columns2 = ['goid', 'node_depth', 'n_classes', 'roc_auc_score', 'evol_time']

    def __init__(self, goid, subclasses, subnodes, depth, aspect) -> None:
        self.goid = goid
        self.subclasses = subclasses
        self.subnodes = subnodes
        self.depth = depth
        self.aspect = aspect

    def to_tsv(self):
        return (self.aspect
                + '\t' + self.goid 
                + '\t' + str(self.depth) 
                + '\t' + ','.join([x.goid for x in self.subnodes]) 
                + '\t' + ','.join(self.subclasses))

    def to_tsv_result(self):
        return (self.aspect
                + '\t' + self.goid 
                + '\t' + str(self.depth)
                + '\t' + str(len(self.subclasses))
                + '\t' + str(self.roc_auc_score)
                + '\t' + str(self.evol_time))

    def create_train_dataset(self, datasets_dir, train_terms_all, 
                             train_protein_ids_all, train_plm_embeddings, max_samples=30000):
        train_terms_updated = train_terms_all[train_terms_all['term'].isin(self.subclasses)]
        train_protein_ids = train_terms_updated['EntryID'].unique().tolist()
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

    def train(self, test_size=0.25, batch_size=5120):
        print('Splitting')
        train_features, train_labels, test_features, test_labels = iterative_train_test_split(self.features, 
            self.train_labels, test_size = test_size)
        #assert len(train_plm_embeddings_train) == len(train_labels_train)
        '''print('Creating model')
        INPUT_SHAPE = [train_features.shape[1]]
        print(train_features.shape)
        BATCH_SIZE = batch_size

        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),    
            tf.keras.layers.Dense(units=train_features.shape[1]*0.75, activation='relu'),
            tf.keras.layers.Dense(units=train_features.shape[1]*0.6, activation='relu'),
            tf.keras.layers.Dense(units=train_features.shape[1]*0.5, activation='relu'),
            tf.keras.layers.Dense(units=len(train_labels[0]),activation='sigmoid')
        ])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy'
        )

        self.history = self.model.fit(
            train_features, train_labels,
            batch_size=BATCH_SIZE,
            epochs=6)
        
        y_pred = self.model.predict(test_features)'''
        '''y_pred_argmax=np.argmax(y_pred, axis=1)
        #y_test_argmax=np.argmax(test_labels, axis=1)
        cat_accuracy = metrics.CategoricalAccuracy()
        cat_accuracy.update_state(test_labels, y_pred)
        self.categorical_acc = cat_accuracy.result().numpy()
        #self.f1 = metrics.f1_score(y_test_argmax, y_pred_argmax, average='samples')
        #self.accuracy = metrics.accuracy_score(test_labels, y_pred)'''
        '''try:
            self.roc_auc_score = metrics.roc_auc_score(test_labels, y_pred)
            self.average_precision_score = metrics.average_precision_score(test_labels, y_pred)
            print(self.roc_auc_score, self.average_precision_score)
            self.failed = False
        except ValueError as err:
            print(err)
            print(test_labels[0])
            print(test_labels.shape)
            self.failed = True'''

        start_time = time()
        try:
            ga = ClassificationOptimizer(train_features, test_features, 
                                    train_labels, test_labels, 
                                    gens=5, pop=20, parents=10)
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
        
    def erase_dataset(self):
        self.features = None
        self.train_labels = None

    def set_path(self, datasets_dir):
        self.path = path.join(datasets_dir, self.goid.lstrip('GO:') + '_node.obj')

def node_factory(node_description_df_path: str) -> List[Node]:
    stream = open(node_description_df_path, 'r')
    new_nodes = []
    header = stream.readline()
    for rawline in stream.readlines():
        cells = rawline.rstrip('\n').split('\t')
        aspect, goid, node_depth, child_nodes, classes = cells
        new_nodes.append(Node(
            goid, classes.split(','), [], int(node_depth), aspect
        ))
    return new_nodes

def count_subclasses(children, children_descendants):
    subclasses = set(children)
    #print(subclasses)
    #print([type(x) for x in children_descendants])
    for c in children_descendants:
        subclasses.update(c)
    return subclasses

def create_node(go_id, classes: dict, graph: MultiDiGraph, 
                id_to_name: dict, aspect: str, ident = 0, max_children = 250):
    print('\t'*ident + 'Making node for', id_to_name[go_id])
    children = list(graph.predecessors(go_id))
    #TODO: len(children) > max_children
    children_descendants = [list(nx.ancestors(graph, child_id)) for child_id in children]
    n_subclasses = len(count_subclasses(children, children_descendants))
    subnodes = []
    max_to_consider = max_children if len(children) < max_children else len(children)
    while n_subclasses > max_children:
        print('\t'*ident + 'Too big', n_subclasses)
        largest_class_index = 0
        for class_index in range(len(children)):
            if len(children_descendants[class_index]) > len(children_descendants[largest_class_index]):
                largest_class_index = class_index
        subnode = create_node(children[largest_class_index], classes, graph, id_to_name, aspect, ident = ident+1, max_children=max_children)
        children_descendants[largest_class_index] = []
        subnodes.append(subnode)
        n_subclasses = len(count_subclasses(children, children_descendants))
        print('\t'*ident + 'Reduced to', n_subclasses)

    all_subclasses = list(count_subclasses(children, children_descendants))
    print('\t'*ident + 'Made node for', id_to_name[go_id], 'with', 
          len(children), '+', [len(x) for x in children_descendants])
    return Node(go_id, all_subclasses, subnodes, ident, aspect)

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
        goid = node.goid
        if not goid in node_by_id:
            node_by_id[goid] = []
        node_by_id[goid].append(node)
    
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