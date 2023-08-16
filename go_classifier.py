import glob
from os import path
import pickle
from tqdm import tqdm

from node import Node


class GoClassifier():

    def __init__(self, models_path, datasets_path, graph) -> None:
        self.classifier_collection = self.load_models(models_path, datasets_path)
        self.graph = graph
        
    def load_models(self, models_path, datasets_path):
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
        print('Loading classification models')
        for node in tqdm(nodes):
            node.load_model()
        return nodes
    
    def classify_with_node(self, feature_list, protein_list, node: Node):
        classifications = node.classify(feature_list, protein_list)

        return classifications
        
    def create_classifications(self, feature_list, protein_list, output_path):
        all_classifications = []
        print('Making classifications')
        for classifier in tqdm(self.classifier_collection):
            all_classifications += self.classify_with_node(feature_list,
                protein_list, classifier)
        all_classifications.sort(
            key = lambda clf: (clf[1], clf[2], clf[0], clf[3]))
        
        output = open(output_path, 'w')
        output.write('CLASSIFIER_NAME\tPROTEIN\tGENE_ONTOLOGY\tPROB\n')
        for clf in all_classifications:
            output.write('\t'.join([str(x) for x in clf]) + '\n')

        return len(all_classifications)