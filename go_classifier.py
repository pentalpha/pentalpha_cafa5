import glob
from os import path
import pickle
from tqdm import tqdm

from node import Node, node_from_json
from post_processing import solve_protein
from prepare_data import chunks

def find_nodes(node_description_df_path: str, 
                models_path: str, datasets_path: str, aspect_to_load: str):
    stream = open(node_description_df_path, 'r')
    nodelist = []
    header = stream.readline()
    for rawline in stream.readlines():
        cells = rawline.rstrip('\n').split('\t')
        aspect, nodename, goid, node_depth, child_nodes, classes = cells
        if aspect == aspect_to_load:
            #print('trying to find', nodename)
            node_path = datasets_path + '/'+nodename.lstrip('GO:')+'_node.json'
            model_path = models_path + '/'+nodename.lstrip('GO:')+'_classifier.json'
            if path.exists(node_path) and path.exists(model_path):
                nodelist.append({'name': nodename, 
                                'depth': int(node_depth),
                                'node_path': node_path,
                                'model_path': model_path})
    nodelist.sort(key=lambda x: x['depth'])
    return nodelist

class GoClassifier():

    def __init__(self, node_description_df_path, models_path, 
                 datasets_path, graph, aspect) -> None:
        self.models_path = models_path
        self.loadable_nodes = find_nodes(node_description_df_path, 
                models_path, datasets_path, aspect)
        self.graph = graph
    
    def set_limit(self, max_nodes=10):
        self.node_lists = list(chunks(self.loadable_nodes, max_nodes))
        print('Separated',  len(self.loadable_nodes), 'nodes in', 
              len(self.node_lists), 'lists')
        for node_list in self.node_lists:
            print('\t', len(node_list))
        
        self.total_nodes = sum([len(l) for l in self.node_lists])

    def load_model(self, loadable):
        node = node_from_json(loadable['node_path'])
        node.set_model_path(self.models_path)
        node.load_model()
        return node
    
    def classify_with_node(self, feature_list, protein_list, node: Node):
        classifications = node.classify(feature_list, protein_list)

        return classifications

    def classify_proteins(self, features, ids, output):
        classifications_final = [[] for protID in ids]
        classifications_buffers = [[] for protID in ids]

        current_depth = 0
        #bar = tqdm(total=self.total_nodes)
        for nodelist in self.node_lists:
            nodes = [self.load_model(loadable) 
                     for loadable in nodelist]
            for node in nodes:
                new_clfs = self.classify_with_node(features, ids, node)
                for i in range(len(ids)):
                    classifications_buffers[i] += new_clfs[i]
                if node.depth > current_depth:
                    for i in range(len(ids)):
                        new_solved = solve_protein(
                            classifications_final[i],
                            classifications_buffers[i],
                            node.depth, ids[i])
                        classifications_final[i] += new_solved
                        classifications_buffers[i] = []
                        current_depth = node.depth
                #bar.update(1)
        #bar.close()

        for solved_clfs in classifications_final:
            for prot_id, goid, prob in solved_clfs:
                output.write(prot_id+'\t'+goid+'\t'+str(prob)+'\n')