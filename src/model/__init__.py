from model.tcf import TCFModel
from model.wearec import WEARecModel


MODEL_DICT = {
    "wearec": WEARecModel,
    "tcf": TCFModel,
}

__all__ = ["MODEL_DICT", "TCFModel", "WEARecModel"]
