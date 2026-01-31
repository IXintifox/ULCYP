import lmdb
import pickle
import tqdm
import yaml
from easydict import EasyDict
# from tools.logger import logger


def load_config(path="Config/vdata.yaml"):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def load_database(data_base="../DataBase/P450inducerNew.mdb"):
    logger.info('Loading database:%s' % data_base.split("/")[-1])
    mdb = lmdb.open(
        data_base,
        map_size=int(0.5*1024*1024*1024),
        subdir=False,
        meminit=False
    )

    record_info_dict = []
    with mdb.begin(write=False) as txn:
        all_inchikey = [itm.decode() for itm in txn.cursor().iternext(values=False)]
        for itm in tqdm.tqdm(all_inchikey, total=len(all_inchikey)):
            values = pickle.loads(txn.get(itm.encode()))
            if values["mark_error"] == 0:
                record_info_dict.append(values)

    mdb.close()
    return record_info_dict

def editable_database():
    data_base = "../DataBase/P450inducerNew.mdb"
    print("load_data: ", data_base)
    mdb = lmdb.open(
        data_base,
        map_size=int(0.5*1024*1024*1024),
        subdir=False,
        meminit=False
    )
    return mdb

def init_new(path):
    logger.info('Init new database:%s' % path)
    mdb = lmdb.open(
        path,
        map_size=int(10*1024*1024*1024),
        subdir=False,
        meminit=False
    )
    return mdb

def load_features_lmdb(data_base="Data/Repr_data_new.mdb"):
    mdb = lmdb.open(
        data_base,
        map_size=int(7*1024*1024*1024),
        subdir=False,
        meminit=False
    )
    return mdb