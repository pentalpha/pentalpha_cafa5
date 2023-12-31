

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
from classification import makeMultiClassifierModel

from metaheuristic_clf import ClassificationOptimizer
from prepare_data import chunks, count_class_frequencies, delete_errors, detect_only_ones, detect_only_zeros, remove_classes

configs = json.load(open("config.json", 'r'))

random.seed(1337)

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
                             test_size, max_samples=30000):
        
        valid_terms = train_terms_all[train_terms_all['EntryID'].isin(train_protein_ids_all)]
        train_terms_updated = valid_terms[valid_terms['term'].isin(self.subclasses)]
        train_protein_ids = train_terms_updated['EntryID'].unique().tolist()

        #Add 5% of falses to dataset
        '''false_protein_ids = [prot for prot in train_protein_ids_all 
                             if not prot in train_protein_ids]
        max_falses = len(train_protein_ids)*0.05
        falses = random.sample(false_protein_ids, int(max_falses))
        train_protein_ids = train_protein_ids + falses'''

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

        print('Splitting train and test')
        a, b, c, d = iterative_train_test_split(
            features, 
            train_labels, test_size = test_size)
        
        self.train_features = a
        self.train_labels = b
        self.test_features = c
        self.test_labels = d

        print('Counting frequencies')
        class_freqs = count_class_frequencies(self.test_labels, self.subclasses)
        unfrequent = []
        for goid, freq in class_freqs.items():
            if freq < 8:
                print(goid, freq)
                unfrequent.append(goid)
        if len(unfrequent) > 0:
            unfrequent_indexes = [goid_to_col[goid] for goid in unfrequent]
            self.train_labels = remove_classes(self.train_labels, unfrequent_indexes)
            self.test_labels = remove_classes(self.test_labels, unfrequent_indexes)

            print('Removing', unfrequent)
            self.subclasses = [x for x in self.subclasses if not x in unfrequent]

            assert len(self.train_labels[0]) == len(self.subclasses), str(len(self.train_labels[0])) + ' ' + str(len(self.subclasses))

        zeros = detect_only_zeros(self.test_labels)
        ones = detect_only_ones(self.test_labels)
        errors = zeros + ones
        errors.sort()

        print(len(errors), 'detected')
        if len(errors) > 0:
            print('Deleting then')
            self.test_labels = delete_errors(errors, self.test_labels)
            self.train_labels = delete_errors(errors, self.train_labels)
            self.train_features = delete_errors(errors, self.train_features)
            self.test_features = delete_errors(errors, self.test_features)

    def test_rock_auc_error(self):
        
        n_classes = len(self.test_labels[0])
        n_samples = len(self.test_features)
        mock_pred = [np.random.choice([0, 1], size=n_classes, p=[.1, .9]) 
            for i in range(n_samples)]
        try:
            roc_auc_score = metrics.roc_auc_score(self.test_labels, mock_pred)
        except ValueError as err:
            print(err)
            print('Examples:')
            print(self.test_labels[10])
            print(self.test_labels[100])
            print(self.test_labels[1000])
            zeros = detect_only_zeros(self.test_labels)
            ones = detect_only_ones(self.test_labels)
            errors = zeros + ones
            errors.sort()
            print(len(errors), 'detected')

            print('Deleting then')
            self.test_labels = delete_errors(errors, self.test_labels)
            mock_pred = delete_errors(errors, mock_pred)

            zeros1 = detect_only_zeros(mock_pred)
            ones1 = detect_only_ones(mock_pred)
            errors1 = zeros1 + ones1
            errors1.sort()
            print(len(errors1), 'detected in pred')

            print('Testing again')
            tries = 100
            while tries > 0:
                try:
                    roc_auc_score = metrics.roc_auc_score(self.test_labels, mock_pred)
                    print('success')
                    return None
                except ValueError as err:
                    tries -= 1
            print('No more tries')

            print('Counting class frequencies')
            

            quit()
            '''print(err)

            sums = [(i, sum(self.test_labels[i])) for i in range(len(self.test_labels))]
            sums.sort(key=lambda x: x[1])

            print('smallest')
            print(sums[0])
            print(self.test_labels[sums[0][0]])

            print('largest')
            print(sums[-1])
            print(self.test_labels[sums[-1][0]])
            quit()'''
            '''for test_i in range(len(self.test_labels)):
                test = self.test_labels[test_i]
                total = sum(test)
                if total == 0:
                    zeros.append(test_i)
                elif total == len(test):
                    ones.append(test_i)
            print(len(zeros), 'only zeros')
            print(len(ones), 'only ones')

            quit()'''

    def train(self):
        start_time = time()
        '''try:
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
            return None'''
        '''param_dict = {'batch_size': 1250, 'learning_rate': 0.0016, 
                      'epochs': 11, 'hidden1': 0.2, 
                      'hidden2': 0.4}'''
        param_dict = {'batch_size': 1070, 'learning_rate': 0.0015, 'epochs': 11, 
                      'hidden1': 0.71, 'hidden2': 0.82}
        optimizer = ClassificationOptimizer.make_optimizer('AdamOptimizer', 
                                                           param_dict['learning_rate'])
        print(param_dict)
        annot_model = makeMultiClassifierModel(
            self.train_features, self.train_labels,
            param_dict['batch_size'],
            [param_dict['hidden1'], param_dict['hidden2']],
            optimizer, 
            param_dict['epochs'])
        self.evol_time = time() - start_time
        self.best_params = param_dict
        print('GA terminando após', self.evol_time/60, 'minutos')
        y_pred = annot_model.predict(self.test_features)
        try:
            self.roc_auc_score = metrics.roc_auc_score(self.test_labels, y_pred)
        except ValueError as err:
            self.failed = True
            print(err)
            print('AOC error at', self.goid, self.name)
            return None
        
        print('score', self.roc_auc_score)
        print("Best Params:", param_dict)
        print('evol_time', self.evol_time)
        print(annot_model.summary())

        if self.roc_auc_score > 0.5:
            json_to_save = annot_model.to_json()
            open(self.model_path, 'w').write(json_to_save)
            annot_model.save_weights(self.weights_path)
            #annot_model.save(self.model_path)
            self.failed = False
        else:
            self.failed = True
    '''ga_info = [ga.solutions_by_generation, ga.fitness_by_generation, 
            ga.evolved, ga.finish_type, self.evol_time]'''
    
    
    
    '''annot_model = ga.reg
    print("Best Params:", ga.best_params)
    print('\n'.join([str(fitvec) for fitvec in ga.fitness_by_generation]))
    print('evolved', ga.evolved)
    print('ga.finish_type', ga.finish_type)
    print('evol_time', self.evol_time)'''

    

    '''self.roc_auc_score = ga.score
    self.best_params = ga.best_params'''
        
    def erase_dataset(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.annot_model = None

    def set_model_path(self, models_dir):
        model_base_path = path.join(models_dir, self.name.lstrip('GO:') + '_classifier.')
        self.model_path = model_base_path+'json'
        self.weights_path = model_base_path+'h5'

    def load_model(self):
        loaded_model = tf.keras.models.model_from_json(
            open(self.model_path, 'r').read())
        loaded_model.load_weights(self.weights_path)
        self.annot_model = loaded_model

    def set_path(self, datasets_dir):
        self.path = path.join(datasets_dir, self.name.lstrip('GO:') + '_node.json')
    
    def to_json_untrained(self):
        data = {
            "aspect": self.aspect,
            "name": self.name,
            "goid": self.goid,
            "depth": self.depth,
            "subclasses": self.subclasses,
            "path": self.path
        }
        json_str = json.dumps(data, indent=2)
        open(self.path, 'w').write(json_str)

    def to_json(self):
        data = {
            "aspect": self.aspect,
            "name": self.name,
            "goid": self.goid,
            "depth": self.depth,
            "subclasses": self.subclasses,
            "roc_auc_score": self.roc_auc_score,
            "evol_time": self.evol_time,
            "model_path": self.model_path,
            "weights_path": self.weights_path,
            "failed": self.failed,
            "best_params": self.best_params,
            "path": self.path
        }
        json_str = json.dumps(data, indent=2)
        open(self.path, 'w').write(json_str)

    def classify(self, input_X, protein_names):
        y_pred = self.annot_model.predict(input_X, verbose = 0)
        assert len(y_pred[0]) == len(self.subclasses)
        predicted = []
        for protein_index in range(len(y_pred)):
            probs_vec = y_pred[protein_index]
            prot_name = protein_names[protein_index]
            prob_tuples = [(self.name, prot_name, 
                            self.subclasses[i], probs_vec[i]) 
                           for i in range(len(probs_vec))]
            predicted.append(prob_tuples)
        
        return predicted

def node_from_json(node_path):
    data = json.loads(open(node_path,'r').read())
    node = Node(
        data['goid'], data['subclasses'], [], data['depth'], data['aspect']
    )
    node.name = data['name']
    node.roc_auc_score = data['roc_auc_score']
    node.evol_time = data['evol_time']
    node.model_path = data['model_path']
    node.failed = data['failed']
    node.best_params = data['best_params']
    node.path = data['path']
    node.weights_path = data['weights_path']

    return node

def node_from_json_untrained(node_path):
    data = json.loads(open(node_path,'r').read())
    node = Node(
        data['goid'], data['subclasses'], [], data['depth'], data['aspect']
    )
    node.name = data['name']
    node.path = data['path']

    return node

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

def load_node_depths(node_description_df_path: str) -> List[Node]:
    stream = open(node_description_df_path, 'r')
    depths = {}
    header = stream.readline()
    for rawline in stream.readlines():
        cells = rawline.rstrip('\n').split('\t')
        aspect, nodename, goid, node_depth, child_nodes, classes = cells
        depths[nodename] = int(node_depth)
    return depths

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