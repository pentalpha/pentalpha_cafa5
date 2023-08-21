import json
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool

from prepare_data import chunks

configs = json.load(open("config.json", 'r'))

def choose_prob(prot_id, goid, preds, solved_probs):
    '''print(prot_id, 'has', len(preds), 'predictions of', goid)
    #print(len(solved_probs), 'current solved of', prot_id)
    for parent_go, prob in preds:
        print('\t', parent_go, prob)'''
    parent_and_prob = [(parent_go.split('_')[0], prob) 
        for parent_go, prob in preds]
    parent_probs = {}
    for parent, _ in parent_and_prob:
        for _, goid, prob in solved_probs:
            if parent == goid:
                parent_probs[parent] = prob
                break
        if not parent in parent_probs:
            parent_probs[parent] = -1
    '''if -1 in parent_probs.values():
        print(parent_probs)
        for _, goid, prob in solved_probs:
            print(goid, prob)
            quit()'''
    parent_and_prob.sort(key = lambda x: parent_probs[x[0]])
    parent_higher, prob_higher = parent_and_prob[-1]
    
    return parent_higher, prob_higher

def solve_protein(current_solved, protein_preds, depth, prot_id):
    #current_solved = [x for x in solved_probs if x[0] == prot_id]
    new_solved = []
    by_goid = {}
    #print(protein_preds[0])
    #print(current_solved[0] if len(current_solved) > 0 else None)
    for clf_name, _, goid, prob in protein_preds:
        if not goid in by_goid:
            by_goid[goid] = []
        by_goid[goid].append([clf_name, prob])

    n_to_solve = 4
    for goid, preds in by_goid.items():
        if len(preds) == 1 or depth == 0:
            pred = preds[0]
            new_solved.append([prot_id, goid, pred[1]])
        else:
            clf_name, prob = choose_prob(prot_id, goid, preds, 
                current_solved)
            new_solved.append([prot_id, goid, prob])
            #quit()

    return new_solved

def solve_protein_chunk(prod_dfs, solved_probs, current_depth):
    new_solveds = []
    for protid, protdf in tqdm(prod_dfs):
        new_solved = solve_protein(solved_probs, protdf, current_depth, protid)
        new_solveds += new_solved
    return new_solveds

def filter_solved(solved_probs, protids):
    print(len(solved_probs), 'solved being filtered to', len(protids), 'proteins')
    filtered_solved_probs = [x for x in solved_probs if x[0] in protids]
    print(len(filtered_solved_probs))
    return filtered_solved_probs

def solve_probabilities(ia_file, go_graph, node_depths, solved_probs, current_depth):
    import pandas as pd
    #depths = sorted(list(set(node_depths.values())))
    #predictions_by_depth = {d: [] for d in depths}
    print('Reading probabilities df')
    preds_df = pd.read_csv(ia_file, sep='\t', index_col=False)
    print(preds_df)
    print(len(preds_df), 'predictions in depth', current_depth)
    print('Grouping by protein')
    prod_dfs = []
    for protid, subdf in tqdm(preds_df.groupby('PROTEIN')):
        prod_dfs.append((protid, subdf))
    
    print('Separating chunks')
    processes = (configs['processes']+2)*2
    df_chunks = list(chunks(prod_dfs, int(len(prod_dfs)/processes)))
    protid_chunks = []
    for prod_dfs in df_chunks:
        protids = [protid for protid, df in prod_dfs]
        protid_chunks.append(protids)
    param_list = [(df_chunks[i], 
                   filter_solved(solved_probs, protid_chunks[i]), 
                   current_depth)
                  for i in range(len(df_chunks))]
    
    print('Solving proteins with', processes, 'processes')
    with Pool(processes) as p:
        new_solveds = p.starmap(solve_protein_chunk, param_list)
        print('Concatenating')
        for new_solved in new_solveds:
            solved_probs += new_solved
    
    '''for protid, protdf in tqdm(prod_dfs):
        new_solved = solve_protein(solved_probs, protdf, current_depth, protid)
        solved_probs += new_solved'''
    '''print('Reading probabilities')
    probs_stream = open(ia_file, 'r')
    header = probs_stream.readline()
    for rawline in probs_stream.readlines():
        clf_name, prot_id, goid, prob_str = rawline.rstrip('\n').split('\t')
        preds.append([clf_name, prot_id, goid, float(prob_str)])
    
    print(len(preds), 'predictions in depth', current_depth)
    protein_ids = set()
    for clf_name, prot_id, goid, prob in preds:
        protein_ids.add(prot_id)
    
    print('Solving proteins')
    for protein_id in tqdm(protein_ids):
        protein_preds = [x for x in preds if x[1] == protein_id]
        new_solved = solve_protein(solved_probs, protein_preds, current_depth)
        solved_probs += new_solved'''
    
    print(len(solved_probs), 'solved predictions after depth', current_depth)
    #return solved_probs