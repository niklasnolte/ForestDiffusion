import DISK.lib as lib
from kbgen.model import KBFormer
from kbgen.config import tabddpm_config as config
from kbgen.utils import mup_model
from DISK.scripts.utils_train import make_dataset
import numpy as np


def get_model_dataset(T_dict, model_params, real_data_path, device, change_val, use_mup):
    """get model, dataset and update config with dataset and model params"""
    T = lib.Transformations(**T_dict)
    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params["num_classes"],
        is_y_cond=model_params["is_y_cond"],
        change_val=change_val,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if T_dict["cat_encoding"] == "one-hot":
        raise NotImplementedError("one-hot encoding not checked yet")

    num_numerical_features = (
        dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    )

    # model params
    config["dropout"] = model_params["rtdl_params"]["dropout"]
    config["num_decoder_mixtures"] = model_params["rtdl_params"]["num_decoder_mixtures"]
    config["d_model"] = model_params["rtdl_params"]["d_model"]
    config["init_var"] = model_params["rtdl_params"].get("init_var", 1.0)
    config["num_emb"] = model_params["rtdl_params"]["num_emb"]

    # task params
    config["num_numerical_features"] = num_numerical_features
    config["num_category_classes"] = K
    config["is_y_cond"] = model_params["is_y_cond"]
    config["num_classes"] = model_params[
        "num_classes"
    ]  # this is the target number of classes

    # init model with mup
    if use_mup:
        model = mup_model(KBFormer, {"d_model": 8}, config)
    else:
        model = KBFormer(config)
    
    model.to(device)
    return model, dataset
