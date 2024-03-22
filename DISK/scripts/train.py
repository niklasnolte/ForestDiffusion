from copy import deepcopy
import tqdm
import torch
import os
import numpy as np
from DISK.scripts.utils_train import update_ema
import DISK.lib as lib
import pandas as pd
from kbgen.diffusion import HybridDiffusion
from DISK.scripts.utils_custom import get_model_dataset
import mup


class Trainer:
    def __init__(
        self,
        diffusion,
        train_iter,
        lr,
        weight_decay,
        steps,
        device=torch.device("cuda:1"),
        use_mup=False,
        verbose=True,
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        AdamW = mup.MuAdamW if use_mup else torch.optim.AdamW
        self.optimizer = AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.device = device
        self.loss_history = pd.DataFrame(columns=["step", "mloss", "gloss", "loss"])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

        self.verbose = verbose

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, y):
        # concat version of all x_cat, x_num and y in the regression case
        x = x.to(self.device)
        # if torch.isnan(x).any():
        # raise ValueError("NaNs in input")
        y = y.long().to(self.device)
        self.optimizer.zero_grad()
        loss = self.diffusion.loss(x, y)

        loss.backward()
        self.optimizer.step()

        return loss, torch.zeros_like(loss)

    def run_loop(self):
        pbar = tqdm.trange(self.steps) if self.verbose else range(self.steps)
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        for step in pbar:
            x, y = next(self.train_iter)
            batch_loss_multi, batch_loss_gauss = self._run_step(x, y)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(
                        f"Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}"
                    )
                self.loss_history.loc[len(self.loss_history)] = [
                    step + 1,
                    mloss,
                    gloss,
                    mloss + gloss,
                ]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion.model.parameters())

            if self.verbose:
                pbar.set_description_str(f"{batch_loss_multi.item()}")


def train(
    parent_dir,
    real_data_path="data/higgs-small",
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    model_type="mlp",
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    device=torch.device("cuda"),
    seed=0,
    change_val=False,
    use_mup=False,
    verbose=True,
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)


    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = lib.Transformations(**T_dict)

    # dataset = make_dataset(
    #     real_data_path,
    #     T,
    #     num_classes=model_params["num_classes"],
    #     is_y_cond=model_params["is_y_cond"],
    #     change_val=change_val,
    # )

    # K = np.array(dataset.get_category_sizes("train"))
    # if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
    #     K = np.array([0])
    # print(K)

    # num_numerical_features = (
    #     dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    # )
    # d_in = np.sum(K) + num_numerical_features
    # model_params["d_in"] = d_in
    # print(d_in)

    # print(model_params)

    # # model = get_model(
    # #     model_type,
    # #     model_params,
    # #     num_numerical_features,
    # #     category_sizes=dataset.get_category_sizes('train')
    # # )

    # config["num_numerical_features"] = num_numerical_features
    # config["num_category_classes"] = K
    # config["is_y_cond"] = model_params["is_y_cond"]
    # config["num_classes"] = model_params["num_classes"] # TODO wtf is this

    model, dataset = get_model_dataset(
        T_dict, model_params, real_data_path, device, change_val, use_mup
    )
    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(
        dataset, split="train", batch_size=batch_size
    )

    # diffusion = GaussianMultinomialDiffusion(
    #     num_classes=K,
    #     num_numerical_features=num_numerical_features,
    #     denoise_fn=model,
    #     gaussian_loss_type=gaussian_loss_type,
    #     num_timesteps=num_timesteps,
    #     scheduler=scheduler,
    #     device=device
    # )
    diffusion = HybridDiffusion(model)
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        use_mup=use_mup,
        verbose=verbose,
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, "loss.csv"), index=False)
    torch.save(diffusion.model.state_dict(), os.path.join(parent_dir, "model.pt"))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, "model_ema.pt"))
