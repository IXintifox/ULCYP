import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
# from tools.logger import logger
import pandas as pd
import joblib
import os

class NormalTools:
    def __init__(self):
        self.vt = VarianceThreshold(threshold=0)
        self.ss = StandardScaler()
        self.origin_feature_name = None
        self.feature_name = None
        self.feature_bit = None
        self.analysis_data_number = None

    def __call__(self, features, features_name):
        self.analysis_data_number = len(features)
        self.origin_feature_name = features_name
        df = pd.DataFrame(np.array(features), columns=features_name)
        self.vt.fit(df)
        features = self.vt.transform(df)
        self.feature_name = df.columns[self.vt.get_support()]
        self.ss.fit(features)
        features = self.ss.transform(features)
        features = pd.DataFrame(features, columns=self.feature_name)
        return features

    def norm(self, features):
        df = pd.DataFrame(np.array(features).reshape(1, -1), columns=self.origin_feature_name)
        feature = self.vt.transform(df)
        feature = self.ss.transform(feature)
        return feature

def get_descriptors(mol_list):
    descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptor_values = []

    if len(mol_list) < 500:
        tqdm_state = True
    else:
        tqdm_state = False

    for mol in tqdm(mol_list, total=len(mol_list), desc="Bit analysis || Get descriptors", disable=tqdm_state):
        descriptor_values.append(calculator.CalcDescriptors(mol))
    return descriptor_values, descriptor_names

def get_descriptors_bits(mol_list):
    logger.info('Getting descriptors bits')
    normal_tools = NormalTools()
    descriptor_values, descriptor_names = get_descriptors(mol_list)
    features = normal_tools(descriptor_values, descriptor_names)
    for i in tqdm(range(100), total=100, desc="Bit analysis || Check des norm"):
        assert (features.iloc[i, :].values == normal_tools.norm(descriptor_values[i])[0]).all()
    return normal_tools

def get_norm_tools(mol):
    finger_tools = get_fingerprint_bits(mol)
    des_tools = get_descriptors_bits(mol)
    return finger_tools, des_tools

def get_fingerprint(mol):
    fp_gen = GetMorganGenerator(radius=2)
    return fp_gen.GetFingerprint(mol)

def get_fingerprint_bits(mol):
    logger.info('Getting fingerprint bits')
    normal_tools = NormalTools()
    fingerprints_cover = [get_fingerprint(itm) for itm in tqdm(mol, total=len(mol),
                                                                     desc="Bit analysis || To ECFP")]

    features = normal_tools(fingerprints_cover, ["fp%s"% i for i in range(len(fingerprints_cover[0]))])
    for i in tqdm(range(100), total=100, desc="Bit analysis || Check finger norm"):
        assert (features.iloc[i, :].values == normal_tools.norm(fingerprints_cover[i])[0]).all()

    return normal_tools

def bit_analysis(smiles, use_cache=True, save_cache=True):
    # smiles = smiles[:2000]
    if use_cache:
        logger.info('Use cache')
        if os.path.exists("tools/norm/norm_cache.tools"):
            logger.info('Cache path:"tools/norm/norm_cache.tools"')
            finger_tools, des_tools = joblib.load("tools/norm/norm_cache.tools")
            if finger_tools.analysis_data_number != len(smiles):
                logger.info("----> Cache not consistent, re-norm")
            else:
                logger.info("----> Cache loaded Successfully")
                return finger_tools, des_tools
        else:
            logger.info("----> Cache file not found, re-norm")
    mol = [Chem.MolFromSmiles(itm) for itm in tqdm(smiles, total=len(smiles), desc="Bit analysis || To mol")]
    finger_tools, des_tools = get_norm_tools(mol)
    if save_cache:
        logger.info('Save cache to tools/norm/norm_cache.tools')
        joblib.dump((finger_tools, des_tools), "tools/norm/norm_cache.tools")
    return finger_tools, des_tools