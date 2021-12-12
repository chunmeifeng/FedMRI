from .defaults import _C as cfg
from .different_dataset_client import _C as DDC
from .different_dataset_different_subsample import _C as DDDC

config_factory = {
    'different_dataset_client': DDC,
    'different_dataset_different_subsample': DDDC,
}

def build_config(factory):
    return config_factory[factory]