import tqdm
from tools.load_database import (load_database,
                                 load_config,
                                 init_new)
from rdkit import Chem
from rdkit.Chem import (MolFromSmiles,
                        MolToInchiKey)
import numpy as np
from networkx import Graph
from tools.bit_choice import (bit_analysis,
                              get_fingerprint,
                              get_descriptors)
# from tools.logger import logger
from tools.graph_tools import MolecularEncoder
from tools.graph_calculate import SmilesUMA2Graph
from tools import bit_choice
import pickle

class Representation:
    def __init__(self, bit_choice=True, use_cache=True, full_smiles=None):
        self.finger_tools = None
        self.des_tools = None
        if bit_choice:
            if full_smiles is None:
                logger.error("Bit choice error")
                raise NotImplementedError
            self.finger_tools, self.des_tools = bit_analysis(full_smiles, use_cache=use_cache)
        self.uma_graph = SmilesUMA2Graph()

    def __call__(self, smiles):
        mol = MolFromSmiles(smiles)
        assert mol is not None
        fingerprint, descriptors = self.get_fingerprint_and_descriptors(mol)
        graph_with_h, features_h, graph_no_h, features_no_h = self.uma_graph(smiles)

        return descriptors, fingerprint, graph_with_h, features_h, graph_no_h, features_no_h

    def get_fingerprint_and_descriptors(self, mol):
        fingerprint = get_fingerprint(mol)
        descriptors, _ = get_descriptors([mol])
        if False:
            fingerprint = self.finger_tools.norm(fingerprint)
            descriptors = self.des_tools.norm(descriptors)
        return fingerprint, descriptors

#
# def task_class_judgement(itm):
#     task_class = load_config("Config/Data.yaml")["task_class"]

def label_vector(itm):
    vect_bit = load_config("Config/vdata.yaml")["vector_bit"]
    vect = []
    for i in vect_bit:
        vect.append(itm[i])
    vect = np.array(vect)
    return vect

def dataset_map():
    all_items = load_database()
    logger.info("Data processing")
    all_smiles = [itm["Smiles"] for itm in all_items]
    init_feature_tools = Representation(False, use_cache=True, full_smiles=all_smiles)
    init_database = init_new("Data/Repr_data_new.mdb")
    with init_database.begin(write=True) as txn:
        for itx, itm in tqdm.tqdm(enumerate(all_items), total=len(all_items), desc="Mapping"):
            smiles = itm['Smiles']
            label = label_vector(itm)
            datasets = itm["dataset"]
            inchi_key = MolToInchiKey(MolFromSmiles(smiles))
            descriptors, fingerprint, graph_with_h, features_h, graph_no_h, features_no_h = init_feature_tools(smiles)

            assert len(datasets) == np.sum(label!=-1), f"error{itx}"

            itm_dict = {
                "smiles": smiles,
                "inchi_key": inchi_key,
                "datasets": datasets,
                "label": label,
                "descriptors": descriptors,
                "fingerprint": fingerprint,
                "graph_with_h": graph_with_h,
                "graph_no_h": graph_no_h,
                "UMA_features_with_h": features_h,
                "UMA_features_no_h": features_no_h,
            }
            txn.put(str(itx).encode(), pickle.dumps(itm_dict))
    logger.info("Data map finished!")

def data_map_for_smiles():
    init_feature_tools = Representation(False, use_cache=True, full_smiles=all_smiles)
    init_database = init_new("Data/Repr_data_new.mdb")
    with init_database.begin(write=True) as txn:
        for itx, itm in tqdm.tqdm(enumerate(all_items), total=len(all_items), desc="Mapping"):
            smiles = itm['Smiles']
            label = label_vector(itm)
            datasets = itm["dataset"]
            inchi_key = MolToInchiKey(MolFromSmiles(smiles))
            descriptors, fingerprint, graph_with_h, features_h, graph_no_h, features_no_h = init_feature_tools(smiles)

            assert len(datasets) == np.sum(label!=-1), f"error{itx}"

            itm_dict = {
                "smiles": smiles,
                "inchi_key": inchi_key,
                "datasets": datasets,
                "label": label,
                "descriptors": descriptors,
                "fingerprint": fingerprint,
                "graph_with_h": graph_with_h,
                "graph_no_h": graph_no_h,
                "UMA_features_with_h": features_h,
                "UMA_features_no_h": features_no_h,
            }
            txn.put(str(itx).encode(), pickle.dumps(itm_dict))
    logger.info("Data map finished!")

if __name__ == '__main__':
    dataset_map()