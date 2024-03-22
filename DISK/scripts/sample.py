import torch
import numpy as np
import os
from DISK.tab_ddpm.utils import FoundNANsError
from DISK.lib import round_columns
from kbgen.diffusion import HybridDiffusion
from DISK.scripts.utils_custom import get_model_dataset


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1] : indices[i]], axis=1)
        t = X[:, indices[i - 1] : indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def sample(
    parent_dir,
    real_data_path="data/higgs-small",
    batch_size=2000,
    num_samples=0,
    model_type="mlp",
    model_params=None,
    model_path=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    disbalance=None,
    device=torch.device("cuda:1"),
    seed=0,
    change_val=False,
    use_mup=False,
    leaps=1,
    temperature=1.0,
):

    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, D = get_model_dataset(
        T_dict, model_params, real_data_path, device, change_val, use_mup
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = HybridDiffusion(model)

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(
        torch.from_numpy(D.y["train"]), return_counts=True
    )
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == "fix":
        empirical_class_dist[0], empirical_class_dist[1] = (
            empirical_class_dist[1],
            empirical_class_dist[0],
        )
        x_gen, y_gen = diffusion.sample_all(
            num_samples,
            y_dist=empirical_class_dist.float(),
            batch_size=batch_size,
            leaps=leaps,
            temperature=temperature,
        )

    elif disbalance == "fill":
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(
                num_samples,
                y_dist=empirical_class_dist.float(),
                batch_size=batch_size,
                leaps=leaps,
                temperature=temperature,
            )
            x_gen.append(x_temp)
            y_gen.append(y_temp)

        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(
            num_samples,
            y_dist=empirical_class_dist.float(),
            batch_size=batch_size,
            leaps=leaps,
            temperature=temperature,
        )

    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.cpu().numpy(), y_gen.cpu().numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###
    # TODO what's the difference between num_numerical_features and num_numerical_features_?
    num_numerical_features_ = D.X_num["train"].shape[1] if D.X_num is not None else 0

    num_numerical_features = num_numerical_features + int(
        D.is_regression and not model_params["is_y_cond"]
    )

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(
            os.path.join(parent_dir, "X_cat_unnorm"), X_gen[:, num_numerical_features:]
        )
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict["cat_encoding"] == "one-hot":
            X_gen[:, num_numerical_features:] = to_good_ohe(
                D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:]
            )
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(
            os.path.join(parent_dir, "X_num_unnorm"), X_gen[:, :num_numerical_features]
        )
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(
            os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True
        )
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params["num_classes"] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, "X_num_train"), X_num)
        print("Saved samples to:", os.path.join(parent_dir, "X_num_train"))
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, "X_cat_train"), X_cat)
        print("Saved samples to:", os.path.join(parent_dir, "X_cat_train"))
    np.save(os.path.join(parent_dir, "y_train"), y_gen)
    print("Saved samples to:", os.path.join(parent_dir, "y_train"))
