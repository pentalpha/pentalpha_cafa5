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
            '''    print('found')
            else:
                print(node_path, model_path, 'dont exist')'''
    nodelist.sort(key=lambda x: x['depth'])
    return nodelist

class GoClassifier():

    def __init__(self, node_description_df_path, models_path, 
                 datasets_path, graph, aspect) -> None:
        self.models_path = models_path
        self.loadable_nodes = find_nodes(node_description_df_path, 
                models_path, datasets_path, aspect)
        '''self.classifier_collection = self.load_models(models_path, 
            datasets_path, aspect)
        self.classifiers_by_depth = {}
        for node in self.classifier_collection:
            d = node.depth
            if not d in self.classifiers_by_depth:
                self.classifiers_by_depth[d] = []
            self.classifiers_by_depth[d].append(node)'''
        self.graph = graph
    
    def set_limit(self, max_nodes=12):
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

    def classify_proteins(self, features, ids, output_path):
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

        output = open(output_path, 'w')
        for solved_clfs in classifications_final:
            for prot_id, goid, prob in solved_clfs:
                output.write(prot_id+'\t'+goid+'\t'+str(prob)+'\n')
        output.close()

    '''for nodelist in self.node_lists:
            nodes = [self.load_mode(loadable) 
                     for loadable in nodelist]
            for i in range(len(ids)):
                current_depth = 0
                X = [features[i]]
                name = [ids[i]]
                classifications = []
                clfs_buffer = []
                for node in nodes:
                    new_clfs = self.classify_with_node(X, name, node)
                    clfs_buffer += new_clfs
                    if node.depth > current_depth:
                        new_solved = solve_protein(classifications, clfs_buffer, 
                            node.depth, name[0])
                        classifications += new_solved
                        clfs_buffer = []
                        current_depth = node.depth'''
        
    '''def load_models(self, models_path, datasets_path, aspect):
        node_basepath = datasets_path + '/*_node.obj'
        node_paths = glob.glob(node_basepath)
        print('Found', len(node_paths), 'nodes')
        node_and_model = [(node_path, 
                           path.join(models_path, 
                            path.basename(node_path).rstrip('_node.obj')+'_classifier.keras'))
                           for node_path in node_paths]
        for x,y in node_and_model:
            print(x,y)
        node_and_model = [(node_path, model_path) 
                          for node_path, model_path in node_and_model 
                          if path.exists(model_path)]
        print('Found', len(node_and_model), 'nodes with models')
        print('Loading classification nodes')
        nodes = [pickle.load(open(node_path, 'rb'))
                for node_path, model_path in tqdm(node_and_model)]
        nodes = [node for node in nodes if node.aspect == aspect]
        print('Found', len(nodes), 'of aspect', aspect)
        print('Loading classification models')
        for node in tqdm(nodes):
            node.load_model()
        return nodes'''
    
    
        
    def create_classifications(self, feature_list, protein_list, output_path):
        total_clfs = 0
        print('Making classifications')
        for depth, classifiers in self.classifiers_by_depth.items():
            output = open(output_path.replace('.tsv', '_'+str(depth)+'.tsv'), 'w')
            output.write('CLASSIFIER_NAME\tPROTEIN\tGENE_ONTOLOGY\tPROB\n')
            print('Depth', depth)
            clfs = []
            for classifier in tqdm(classifiers):
                new_clfs = self.classify_with_node(feature_list,
                    protein_list, classifier)
                total_clfs += len(new_clfs)
                clfs += new_clfs
            clfs.sort(
                key = lambda clf: (clf[1], clf[2], clf[0], clf[3]))
            for clf in clfs:
                output.write('\t'.join([str(x) for x in clf]) + '\n')
            output.close()
        return total_clfs