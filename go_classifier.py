import glob
from os import path
import pickle
from tqdm import tqdm

from node import Node


class GoClassifier():

    def __init__(self, models_path, datasets_path, graph, aspect) -> None:
        self.classifier_collection = self.load_models(models_path, 
            datasets_path, aspect)
        self.classifiers_by_depth = {}
        for node in self.classifier_collection:
            d = node.depth
            if not d in self.classifiers_by_depth:
                self.classifiers_by_depth[d] = []
            self.classifiers_by_depth[d].append(node)
        self.graph = graph
        
    def load_models(self, models_path, datasets_path, aspect):
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
        return nodes
    
    def classify_with_node(self, feature_list, protein_list, node: Node):
        classifications = node.classify(feature_list, protein_list)

        return classifications
        
    def create_classifications(self, feature_list, protein_list, output_path):
        total_clfs = 0
        print('Making classifications')
        for depth, classifiers in self.classifiers_by_depth.items():
            output = open(output_path.replace('.tsv', '_'+str(depth)+'.tsv'), 'w')
            output.write('CLASSIFIER_NAME\tPROTEIN\tGENE_ONTOLOGY\tPROB\n')
            print('Depth', depth)
            for classifier in tqdm(classifiers):
                new_clfs = self.classify_with_node(feature_list,
                    protein_list, classifier)
                total_clfs += len(new_clfs)
                new_clfs.sort(
                    key = lambda clf: (clf[1], clf[2], clf[0], clf[3]))
                for clf in new_clfs:
                    output.write('\t'.join([str(x) for x in clf]) + '\n')
            output.close()
        return total_clfs