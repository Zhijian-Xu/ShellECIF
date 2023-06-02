import os
from glob import glob
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm
import sys
from typing import Dict

def load_label_map():
    with open('/home/pl/PDBBind/v2020-other-PL/index/INDEX_general_PL_data.2020') as f:
        lines = f.readlines()
    affinity: Dict[str, str] = {}
    for line in lines:
        if line[0] != '#':
            affinity[line.split()[0]] = line.split()[3]

    return affinity

def get_core_pdbids():
    base_path = '/home/pl/PDBBind/CASF-2016/coreset'
    core_dirs = glob(os.path.join(base_path, '*'))
    core_id = set([os.path.split(i)[-1] for i in core_dirs])
    return core_id

def get_pdbids_from_dir(path):
    dirs = glob(os.path.join(path, '*'))
    ids = set([os.path.split(i)[-1] for i in dirs])
    return ids

def get_all_pdbids():
    with open('./pdbids.txt') as f:
        pdbids = f.readline().strip().split(', ')
    return set(pdbids)

pdbids_all = get_all_pdbids()
core_pdbids = get_core_pdbids()
refined2016 = get_pdbids_from_dir('/home/pl/PDBBind/refined-2016/')
refined2019 = get_pdbids_from_dir('/home/pl/PDBBind/refined-2019/')
# refined2020 = get_pdbids_from_dir('/home/pl/PDBBind/refined-2020/')
general2019 = get_pdbids_from_dir('/home/pl/PDBBind/v2019-other-PL/')
# general2020 = get_pdbids_from_dir('/home/pl/PDBBind/v2020-other-PL/')

general_9299 = []
refined_9299 = []

for id in pdbids_all:
    if id in general2019:
        general_9299.append(id)
    else:
        refined_9299.append(id)

# %%
all_label_map = load_label_map()
# train_ids = get_9299_pdbids()
train_ids = set(refined_9299[1000:] + general_9299)
validation_ids = set(refined_9299[:1000])
test_ids = get_core_pdbids()
# refined_ids = get_refined_pdbids()
# train_ids = train_ids - test_ids
if 'labels' in test_ids:
    test_ids.remove('labels')
print('train_ids:', len(train_ids))
print('val_ids', len(validation_ids))
print('test_ids', len(test_ids))

if len(sys.argv) > 1:
    SHELLS = int(sys.argv[1])
else:
    SHELLS = 23
FEATURE_LEN = 1540

print('SHELLS:', SHELLS)

from ECIF.ecif import LoadPDBasDF, LoadSDFasDF, PossibleECIF
from itertools import product
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np

test_path = '/home/pl/PDBBind/CASF-2016/coreset/'
# refined_path = '/home/pl/PDBBind/refined-set/'
refined_2016_path = '/home/pl/PDBBind/refined-2016'
refined_2019_path = '/home/pl/PDBBind/refined-2019'
general_path = '/home/pl/PDBBind/v2019-other-PL/'

def generate_single_ecif_feature(pdbid, distance_cutoff=6.0):
    if pdbid in test_ids:
        proteinfile = os.path.join(test_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(test_path, pdbid, f'{pdbid}_ligand.sdf')
    elif pdbid in refined2016:
        proteinfile = os.path.join(refined_2016_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(refined_2016_path, pdbid, f'{pdbid}_ligand.sdf')
    elif pdbid in refined2019:
        proteinfile = os.path.join(refined_2019_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(refined_2019_path, pdbid, f'{pdbid}_ligand.sdf')
    else:
        proteinfile = os.path.join(general_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(general_path, pdbid, f'{pdbid}_ligand.sdf')

    if not os.path.isfile(proteinfile) or not os.path.isfile(ligandfile):
        print(f'{pdbid} file not found')
        return pdbid, None
    Target = LoadPDBasDF(proteinfile)
    Ligand = LoadSDFasDF(ligandfile)

    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    # Get all possible pairs
    Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    Pairs = pd.concat([Pairs,Distances], axis=1)
    Pairs: pd.DataFrame = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    # Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
    Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
    feature = [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]
    return np.array(feature, dtype=np.float32), all_label_map[pdbid]

def generate_onion_ecif_feature(pdbid, shells=SHELLS):
    if pdbid in test_ids:
        proteinfile = os.path.join(test_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(test_path, pdbid, f'{pdbid}_ligand.sdf')
    elif pdbid in refined2016:
        proteinfile = os.path.join(refined_2016_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(refined_2016_path, pdbid, f'{pdbid}_ligand.sdf')
    elif pdbid in refined2019:
        proteinfile = os.path.join(refined_2019_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(refined_2019_path, pdbid, f'{pdbid}_ligand.sdf')
    else:
        proteinfile = os.path.join(general_path, pdbid, f'{pdbid}_protein.pdb')
        ligandfile = os.path.join(general_path, pdbid, f'{pdbid}_ligand.sdf')

    if not os.path.isfile(proteinfile) or not os.path.isfile(ligandfile):
        print(f'{pdbid} file not found')
        return pdbid, None

    outermost = 0.5 * (shells + 1)
    ncutoffs = np.linspace(1.0, outermost, shells)

    # Avoiding duplicate reading of PDB and SDF files
    # Load 3D file
    Target_origin = LoadPDBasDF(proteinfile)
    Ligand = LoadSDFasDF(ligandfile)

    def GetFeaturesByCutoff(distance_cutoff=6.0):
        Target = Target_origin.copy(deep=True)
        # Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
        for i in ["X","Y","Z"]:
            Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
            Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
        
        # Get all possible pairs
        Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
        Pairs = [x[0]+"-"+x[1] for x in Pairs]
        Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
        Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
        Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
        Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

        Pairs = pd.concat([Pairs,Distances], axis=1)
        Pairs = Pairs[Pairs["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
        # Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
        Pairs["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs["ECIF_PAIR"]]
        return [list(Pairs["ECIF_PAIR"]).count(x) for x in PossibleECIF]

    feature = []

    try:
        for cutoff in ncutoffs:
            one_feature = GetFeaturesByCutoff(cutoff)
            feature.append(np.array(one_feature))
    except AttributeError:
        print(f'[{pdbid}] NoneType object has no attribute "UpdatePropertyCache"')
        return pdbid, None

    for i in range(len(feature)-1, 0, -1):
        feature[i] = feature[i] - feature[i-1]

    return np.array(feature, dtype=np.float32), all_label_map[pdbid]


#-------------- test-feature --------------
test_features = np.zeros((len(test_ids), SHELLS, FEATURE_LEN), dtype=np.float32)
test_labels = np.zeros(len(test_ids), dtype=np.float32)
index = 0

with multiprocessing.Pool(multiprocessing.cpu_count()) as workers:
    with tqdm(total=len(test_ids), ncols=80) as pbar:
        for feature, label in workers.imap_unordered(generate_onion_ecif_feature, test_ids):
            if feature is None or label is None:
                print(f'catch {feature}, {label}')
                continue
            test_features[index] = feature
            test_labels[index] = label
            index += 1
            pbar.update()

folderpath = f'./features_shells_{SHELLS}'
if not os.path.exists(folderpath):
    os.makedirs(folderpath)

print('save as test_ECIF_features.pkl')
with open(f'./features_shells_{SHELLS}/x_test.pkl', 'wb') as f:
    pickle.dump(test_features[:index], f)
with open(f'./features_shells_{SHELLS}/y_test.pkl', 'wb') as f:
    pickle.dump(test_labels[:index], f)

print('featurize test set finished.')

#-------------- val-feature --------------
# with shells
val_features = np.zeros((len(validation_ids), SHELLS, FEATURE_LEN), dtype=np.float32)
val_labels = np.zeros(len(validation_ids), dtype=np.float32)

# without shells
# val_features = np.zeros((len(validation_ids), FEATURE_LEN), dtype=np.float32)
# val_labels = np.zeros(len(validation_ids), dtype=np.float32)
index = 0

print('start featurize validation set...')

with multiprocessing.Pool(multiprocessing.cpu_count()) as workers:
    with tqdm(total=len(validation_ids), ncols=80) as pbar:
        for feature, label in workers.imap_unordered(generate_onion_ecif_feature, validation_ids):
            if feature is None or label is None:
                print(f'catch {feature}, {label}')
                continue
            val_features[index] = feature
            val_labels[index] = label
            index += 1
            pbar.update()

# print('save as validation_onionECIF_features.pkl')
print('save as validation_ECIF_features.pkl')
with open(f'./features_shells_{SHELLS}/x_validation.pkl', 'wb') as f:
    pickle.dump(val_features[:index], f)

# print('save as validation_onionECIF_labels.pkl')
print('save as validation_ECIF_labels.pkl')
with open(f'./features_shells_{SHELLS}/y_validation.pkl', 'wb') as f:
    pickle.dump(val_labels[:index], f)

print('featurize validation set finished.')
print('index:', index)
print('val_features[:index].shape:', val_features[:index].shape)
print('val_labels[:index].shape:', val_labels[:index].shape)

#-------------- train feature --------------
# with shells
train_features = np.zeros((len(train_ids), SHELLS, FEATURE_LEN), dtype=np.float32)
train_labels = np.zeros(len(train_ids), dtype=np.float32)

# without shells
# train_features = np.zeros((len(train_ids), FEATURE_LEN), dtype=np.float32)
# train_labels = np.zeros(len(train_ids), dtype=np.float32)
index = 0

print('start featurize training set...')

with multiprocessing.Pool(multiprocessing.cpu_count()) as workers:
    with tqdm(total=len(train_ids), ncols=80) as pbar:
        for feature, label in workers.imap_unordered(generate_onion_ecif_feature, train_ids):
            if feature is None or label is None:
                print(f'catch {feature}, {label}')
                continue
            train_features[index] = feature
            train_labels[index] = label
            index += 1
            pbar.update()


print('featurize train set finished.')
print('index:', index)
print('train_features[:index].shape:', train_features[:index].shape)
print('train_labels[:index].shape:', train_labels[:index].shape)

print('save as train_ECIF_features.pkl')
with open(f'./features_shells_{SHELLS}/x_train.pkl', 'wb') as f:
    pickle.dump(train_features[:index], f)

print('save as train_ECIF_labels.pkl')
with open(f'./features_shells_{SHELLS}/y_train.pkl', 'wb') as f:
    pickle.dump(train_labels[:index], f)

