import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path, mkdir
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gzip
from lxml import etree
import re
from tqdm import tqdm
import pickle
import sys
from time import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from metaheuristic import RegressionOptimizer

from prepare_data import *
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
#Inputs
xml_path = 'interpro/interpro.xml'

#Outputs
interpro_embeds_path = 'interpro_embeddings.obj'
annot_prot_indexes_path = "annot_prot_indexes_path.obj"
annot_features_path = "annot_features_path.npy"
annot_labels_path = "annot_labels_path.npy"

def create_interpro_embeddings():
    print('create_interpro_embeddings')
    _, used_categories = load_categories_interpro()
    xml_path = 'interpro/interpro.xml'
    xmls = {}

    start_str = "<interpro "
    end_str = "</interpro>"

    current_xml = ""
    started = False
    xml_input = open(xml_path, 'r')
    header_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<interprodb>\n'.encode()
    tail_xml = '</interprodb>'.encode()
    for rawline in xml_input:
        if start_str in rawline:
            started = True
            current_xml = rawline
        elif end_str in rawline:
            current_xml += rawline
            content = header_xml.decode()+current_xml+tail_xml.decode()
            
            data = header_xml + current_xml.encode() + tail_xml
            xml = etree.XML(data)
            all_elements = list(xml.iter())
            #print([element.tag for element in all_elements])
            final_text = ""
            has_go = False
            interproid = None
            for element in all_elements:
                if element.tag == 'abstract':
                    paragraphs = [e for e in all_elements if e.tag == 'p']
                    for p in paragraphs:
                        new = etree.tostring(p).decode().lstrip('<p>').rstrip().rstrip('</p>')
                        new = re.sub("[\(\[].*?[\)\]]", '', new.replace('\n', ''))
                        final_text += new
                    if len(paragraphs) == 0:
                        new = etree.tostring(element).decode().lstrip('<abstract>').rstrip().rstrip('</abstract>')
                        new = re.sub("[\(\[].*?[\)\]]", '', new.replace('\n', ''))
                        final_text += new
                elif element.tag == 'classification':
                    descriptions = [e.text for e in element.iter() if e.tag == 'description']
                    #print(descriptions[0].text)
                    goid = element.attrib.get('id')
                    #print(goid)
                    final_text += '\n'+goid+': '+'. '.join(descriptions)
                    has_go = True
                elif element.tag == 'interpro':
                    interproid = element.attrib.get('id')
            
            if interproid in used_categories:
                xmls[interproid] = final_text

        else:
            if started:
                current_xml += rawline
    
    #print('Writing output texts')
    #dump(xmls, open(xml_filtered, 'w'), indent=4)

    tagged_docs = []

    print("Creating corpus")
    for interproid, full_descript in tqdm(xmls.items()):
        phrase = full_descript.replace(';', '; ').split()
        goids = [x for x in phrase if x.startswith('GO:')]
        new = TaggedDocument(phrase, [interproid])
        tagged_docs.append(new)

    '''print("Training on corpus")
    interpro_embedding = Doc2Vec(tagged_docs, vector_size=128, window=3, min_count=1, workers=4)

    print("Getting separate embeddings for each interproid")
    interpro_embedded = {interproid: list(interpro_embedding.infer_vector(
                            full_descript.replace(';', '; ').split()))
                        for interproid, full_descript in tqdm(xmls.items())}'''
    
    import tensorflow_hub as hub
    use_module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use_model = hub.load(use_module_url)
    print ("module %s loaded" % use_module_url)
    def embed(input):
        return use_model([input])[0]
    '''interpro_embedded = {interproid: embed(full_descript.replace(';', '; ')).numpy().tolist()
                        for interproid, full_descript in tqdm(xmls.items())}
    first = list(interpro_embedded.values())[0]
    print(len(first))
    print(first)
    #dump(interpro_embedded, open(interpro_embeds_path, 'w'), indent=4)
    pickle.dump(interpro_embedded, open(interpro_embeds_path, 'wb'))'''

    _, train_protein_ids, train_plm_embeddings, _ = load_train()
    cat_of_proteins, _ = load_categories_interpro()
    annot_prot_indexes = [prot_i for prot_i in tqdm(range(len(train_protein_ids))) 
                          if train_protein_ids[prot_i] in cat_of_proteins]
    annot_prot_indexes = [prot_i for prot_i in annot_prot_indexes 
                          if len(cat_of_proteins[train_protein_ids[prot_i]]) > 0]
    np.random.seed(1337)
    np.random.shuffle(annot_prot_indexes)
    annot_labels = []
    annot_features = []
    for prot_i in tqdm(annot_prot_indexes):
        protid = train_protein_ids[prot_i]
        
        text_full = '\n'.join([xmls[cat] 
            for cat in cat_of_proteins[protid]]).replace(';', '; ')
        annot_labels.append(embed(text_full).numpy())
        annot_features.append(train_plm_embeddings[prot_i])
    
    annot_features = np.asarray(annot_features)
    annot_labels = np.asarray(annot_labels)
    np.save(open(annot_features_path, 'wb'), annot_features)
    np.save(open(annot_labels_path, 'wb'), annot_labels)
    pickle.dump(annot_prot_indexes, open(annot_prot_indexes_path, 'wb'))

############################################################################

#annot_features_path2 = "annot_features_small.npy"
annot_labels_path2 = "annot_labels_small.npy"
def feature_selection():
    print('Selecting features')
    #annot_features = np.load(annot_features_path)
    annot_labels = np.load(annot_labels_path)

    from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold

    #print('Finding mutual info')
    #annot_labels_small = SelectKBest(mutual_info_regression, k=400).fit_transform(annot_labels, annot_features)
    print("Filtering by variance")
    var_th = VarianceThreshold(threshold=0.0005)
    annot_labels_small = var_th.fit_transform(annot_labels)
    print(annot_labels_small.shape)
    print('Saving results')
    np.save(open(annot_labels_path2, 'wb'), annot_labels_small)

############################################################################
#Inputs:
#annot_features_path
#annot_labels_path

#Outputs:
annot_model_path = "annot_model.obj"

def create_annot_model():
    print('create_annot_model')
    annot_features = np.load(annot_features_path)
    annot_labels = np.load(annot_labels_path2)

    annot_features_train, annot_features_val, annot_labels_train, annot_labels_val = train_test_split(
        annot_features, annot_labels, test_size=0.5, random_state=1337)
    annot_features_val, _, annot_labels_val, _ = train_test_split(
        annot_features_val, annot_labels_val, test_size=0.5, random_state=1337)

    start_time = time()
    ga = RegressionOptimizer(annot_features_train, annot_features_val, 
                             annot_labels_train, annot_labels_val, gens=1, pop=10, parents=5)
    ga.run_ga()
    evol_time = time() - start_time
    print('GA terminando após', evol_time/60, 'minutos')
    ga_info = [ga.solutions_by_generation, ga.fitness_by_generation, 
            ga.evolved, ga.finish_type, evol_time]
    annot_model = ga.reg
    print("Best Params:", ga.genome_to_params(ga.best_params))
    print('\n'.join([str(fitvec) for fitvec in ga.fitness_by_generation]))
    print('evolved', ga.evolved)
    print('ga.finish_type', ga.finish_type)
    print('evol_time', evol_time)

    print(annot_model.summary())
    pickle.dump(annot_model, open(annot_model_path, 'wb'))

############################################################################
#Inputs
#annot_model_path
#train_plm_embeddings_path

#Outputs:
final_annot_features_path = "final_annot_features.npy"

def create_final_annot_features():
    print('create_final_annot_features')
    annot_model = pickle.load(open(annot_model_path, 'rb'))
    _, _, train_plm_embeddings, _ = load_train()
    print('Predicting')
    predicted = annot_model.predict(train_plm_embeddings, workers=4, 
                                    use_multiprocessing=True)
    print(predicted.shape)
    print(predicted[0])
    print('Saving')
    np.save(open(final_annot_features_path, 'wb'), predicted)

############################################################################
#Inputs
#train_terms_path = 'cafa-5-protein-function-prediction/Train/train_terms.tsv'
#train_protein_ids_path = 't5embeds/train_ids.npy'

#Outputs
all_labels_file_path = 'go_labels_train.npy'
labels_mf_file_path = 'go_labels_mf_train.npy'
labels_cc_file_path = 'go_labels_cc_train.npy'
labels_bp_file_path = 'go_labels_bp_train.npy'

def create_clf_labels():
    print('create_clf_labels')
    train_terms_all, train_protein_ids, _, _ = load_train()
    num_of_labels = 1000
    mf = train_terms_all[train_terms_all['aspect'] == 'MFO']
    cc = train_terms_all[train_terms_all['aspect'] == 'CCO']
    bp = train_terms_all[train_terms_all['aspect'] == 'BPO']

    aspect_dfs = [('BP', bp, labels_bp_file_path), 
                  ('MF', mf, labels_mf_file_path), 
                  ('CC', cc, labels_cc_file_path)]

    for aspect_code, train_terms, labels_file_path in aspect_dfs:
        print('Making', aspect_code, 'labels')
        # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
        labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()

        # Fetch the train_terms data for the relevant labels only
        train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]

        print(len(train_terms), len(train_terms_updated), len(train_terms_updated) / len(train_terms))
        train_size = train_protein_ids.shape[0]

        protein_name_to_line = {train_protein_ids[i]: i for i in range(len(train_protein_ids))}
        goid_to_col          = {labels[i]: i for i in range(len(labels))}
        train_labels = [np.array([0.0]*len(labels), dtype=np.float32) for x in train_protein_ids]
        #sub_df = train_terms_updated.head(1000)
        last_prot  = None
        go_indexes = []

        for index, row in tqdm(train_terms_updated.iterrows()):
            prot = protein_name_to_line[row['EntryID']]
            goid = goid_to_col[row['term']]

            if prot != last_prot and last_prot:
                #go_indexes.sort()
                #for go_index in go_indexes:
                #    #train_labels[last_prot][go_index] = 1.0
                train_labels[last_prot] = np.array([1.0 if i in go_indexes else 0.0 for i in range(len(labels))])
                go_indexes = []
                #print(last_prot, train_labels[last_prot][:20])
            go_indexes.append(goid)
            last_prot = prot
        #print(last_prot, go_indexes)
        for go_index in go_indexes:
            train_labels[last_prot][go_index] = 1.0
        train_labels[last_prot] = np.array([1.0 if i in go_indexes else 0.0 for i in range(len(labels))])
        #print(last_prot, train_labels[last_prot][:20])
        go_indexes = []
        f = open(labels_file_path, 'wb')
        np_array = np.asarray(train_labels, dtype=np.float32)
        np.save(f, np_array)
        f.close()

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
        create_interpro_embeddings,
        feature_selection,
        create_annot_model,
        create_final_annot_features,
        create_clf_labels,
        train_clf
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