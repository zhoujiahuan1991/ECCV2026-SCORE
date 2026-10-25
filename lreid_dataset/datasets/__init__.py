from __future__ import absolute_import
import warnings
from .grid import IncrementalSamples4grid
from .prid import IncrementalSamples4prid
from .ilids import IncrementalSamples4ilids
from .viper import IncrementalSamples4viper
from .cuhk01 import IncrementalSamples4cuhk01
from .cuhk02 import IncrementalSamples4cuhk02
from .cuhk03 import IncrementalSamples4cuhk03
from .msmt17 import IncrementalSamples4msmt17
from .sensereid import IncrementalSamples4sensereid
from .market1501 import IncrementalSamples4market
from .dukemtmcreid import IncrementalSamples4duke
from .cuhksysu import IncrementalSamples4subcuhksysu
# from .mix import IncrementalSamples4mix
# from .dc01 import IncrementalSamples4dc01
# from .dc02 import IncrementalSamples4dc02
# from .dc03 import IncrementalSamples4dc03
# from .dc04 import IncrementalSamples4dc04
# from .dc05 import IncrementalSamples4dc05
# from .nkup01 import IncrementalSamples4nkup01
# from .nkup02 import IncrementalSamples4nkup02
# from .nkup03 import IncrementalSamples4nkup03
# from .nkup04 import IncrementalSamples4nkup04
# from .nkup05 import IncrementalSamples4nkup05
# from .ltcc01 import IncrementalSamples4ltcc01
# from .ltcc02 import IncrementalSamples4ltcc02
# from .ltcc03 import IncrementalSamples4ltcc03
# from .ltcc04 import IncrementalSamples4ltcc04
# from .ltcc05 import IncrementalSamples4ltcc05
from .ltcc import IncrementalSamples4ltcc
from .ltcc_cc import IncrementalSamples4ltcc_cc
from .ltcc_gen import IncrementalSamples4ltcc_gen
from .nkup_cc import IncrementalSamples4nkup_cc
from .nkup_sc import IncrementalSamples4nkup_sc
from .prcc import IncrementalSamples4prcc
from .prcc_cc import IncrementalSamples4prcc_cc
from .prcc_sc import IncrementalSamples4prcc_sc
# from .dukemtmc import DukeMTMC
# from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .cuhk03 import CUHK03
# from .cuhk01 import CUHK01
# from .cuhk_sysu import CUHK_SYSU
# from .grid import GRID
# from .sensereid import SenseReID
# from .uav_test import UAV_test
# from .uavhuman import UAVhuman
__factory = {
    'market1501': IncrementalSamples4market,
    'dukemtmc': IncrementalSamples4duke,
    'msmt17': IncrementalSamples4msmt17,
    'cuhk_sysu': IncrementalSamples4subcuhksysu,
    'cuhk03': IncrementalSamples4cuhk03,
    'cuhk01': IncrementalSamples4cuhk01,
    'grid': IncrementalSamples4grid,
    'sense': IncrementalSamples4sensereid,
    # 'mix': IncrementalSamples4mix,
    'ilids':IncrementalSamples4ilids,
    'viper':IncrementalSamples4viper,
    "prid":IncrementalSamples4prid,
    "cuhk02":IncrementalSamples4cuhk02,
    # "dc01":IncrementalSamples4dc01,
    # "dc02":IncrementalSamples4dc02,
    # "dc03":IncrementalSamples4dc03,
    # "dc04":IncrementalSamples4dc04,
    # "dc05":IncrementalSamples4dc05,
    # "nkup01":IncrementalSamples4nkup01,
    # "nkup02":IncrementalSamples4nkup02,
    # "nkup03":IncrementalSamples4nkup03,
    # "nkup04":IncrementalSamples4nkup04,
    # "nkup05":IncrementalSamples4nkup05,
    # "ltcc01":IncrementalSamples4ltcc01,
    # "ltcc02":IncrementalSamples4ltcc02,
    # "ltcc03":IncrementalSamples4ltcc03,
    # "ltcc04":IncrementalSamples4ltcc04,
    # "ltcc05":IncrementalSamples4ltcc05,
    "ltcc":IncrementalSamples4ltcc,
    "ltcc_cc":IncrementalSamples4ltcc_cc,
    "ltcc_gen":IncrementalSamples4ltcc_gen,
    "nkup_cc":IncrementalSamples4nkup_cc,
    "nkup_sc":IncrementalSamples4nkup_sc,
    "prcc":IncrementalSamples4prcc,
    "prcc_cc":IncrementalSamples4prcc_cc,
    "prcc_sc":IncrementalSamples4prcc_sc,
    # 'uav_test':UAV_test,
    # 'uavhuman':UAVhuman
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
