import os
import torch
from scripts.img.opt import get_opts

# data
from torch.utils.data import DataLoader
from datasets.img.imager import ImageDataset
from scripts.img.common import read_image

# models
import commentjson as json
from models.networks.img.NFFB_2d import NFFB

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import StepLR


# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


from utils import load_ckpt, seed_everything, process_batch_in_chunks

# output
import time
from scripts.img.utils import write_image


import warnings; warnings.filterwarnings("ignore")


class ImageSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.time = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

        exp_dir = os.path.join(self.hparams.output_dir, self.time)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)

        ### Load the configuration file
        with open(self.hparams.config) as config_file:
            self.config = json.load(config_file)

        ### Save the configuration file
        path = f"{exp_dir}/config.json"
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4, separators=(", ", ": "), sort_keys=True)

        self.img_data = torch.from_numpy(read_image(self.hparams.input_path)).float()


    def setup(self, stage):
        self.model = NFFB(self.config["network"], out_dims=self.img_data.shape[2])

        ema_decay = 0.95
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            ema_decay * averaged_model_parameter + (1-ema_decay) * model_parameter
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, avg_fn=ema_avg)


        self.train_dataset = ImageDataset(data=self.img_data,
                                          size=1000,
                                          num_samples=self.hparams.batch_size,
                                          split='train')

        self.test_dataset = ImageDataset(data=self.img_data,
                                         size=1,
                                         num_samples=self.hparams.batch_size,
                                         split='test')


    def forward(self, batch):
        b_pos = batch["points"]

        pred = self.model(b_pos)

        return pred


    def on_fit_start(self):
        seed_everything(self.hparams.seed)


    def configure_optimizers(self):
        load_ckpt(self.model, self.hparams.ckpt_path)

        opts = []
        net_params = self.model.get_params(self.config["training"]["LR_scheduler"])
        self.net_opt = FusedAdam(net_params, betas=(0.9, 0.99), eps=1e-15)
        opts += [self.net_opt]

        lr_interval = self.config["training"]["LR_scheduler"][0]["interval"]
        lr_factor = self.config["training"]["LR_scheduler"][0]["factor"]

        if self.config["training"]["LR_scheduler"][0]["type"] == "Step":
            net_sch = StepLR(self.net_opt, step_size=lr_interval, gamma=lr_factor)
        else:
            net_sch = None

        return opts, [net_sch]


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)


    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)


    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)


    def training_step(self, batch, batch_nb, *args):
        results = self(batch)

        b_occ = batch['rgbs'].to(results.dtype)

        batch_loss = (results - b_occ)**2 / (b_occ.detach()**2 + 1e-2)
        loss = batch_loss.mean()

        self.log('lr/network', self.net_opt.param_groups[0]['lr'], True)
        self.log('train/loss', loss)

        return loss


    def training_epoch_end(self, training_step_outputs):
        for name, cur_para in self.model.named_parameters():
            if len(cur_para) == 0:
                print(f"The len of parameter {name} is 0 at epoch {self.current_epoch}.")
                continue

            if cur_para is not None and cur_para.requires_grad and cur_para.grad is not None:
                para_norm = torch.norm(cur_para.grad.detach(), 2)
                self.log('Grad/%s_norm' % name, para_norm)


    def on_before_zero_grad(self, optimizer):
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)


    def backward(self, loss, optimizer, optimizer_idx):
        # do a custom way of backward to retain graph
        loss.backward(retain_graph=True)


    def on_train_start(self):
        gt_img = self.img_data.reshape(self.img_data.shape).float().clamp(0.0, 1.0)
        gt_img = gt_img.cpu().numpy()

        img_path = f'{self.hparams.output_dir}/{self.time}/reference.jpg'
        write_image(img_path, gt_img)
        print(f"\nWriting '{img_path}'... ", end="")


        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log("misc/model_size", model_size)
        print(f"\nThe model size: {model_size}")


    def on_train_end(self):
        # The final validation will use the ema model, as it replaces our normal model
        if self.ema_model is not None:
            print("Replacing the standard model with the EMA model for last validation run")
            self.model = self.ema_model


    def on_validation_start(self):
        torch.cuda.empty_cache()

        if not self.hparams.no_save_test:
            self.val_dir = f'{self.hparams.output_dir}/{self.time}/validation/'
            os.makedirs(self.val_dir, exist_ok=True)


    def validation_step(self, batch, batch_nb):
        img_size = self.img_data.shape[0] * self.img_data.shape[1]

        pred_img = process_batch_in_chunks(batch["points"], self.ema_model, max_chunk_size=2**18)
        pred_img = pred_img[:img_size, :].reshape(self.img_data.shape).float().clamp(0.0, 1.0)

        pred_img = pred_img.cpu().numpy()

        if not self.hparams.no_save_test:
            img_path = f"{self.val_dir}/{self.current_epoch}.jpg"
            write_image(img_path, pred_img)


    def predict_step(self, batch, batch_idx):
        img_size = self.img_data.shape[0] * self.img_data.shape[1]

        pred_img = process_batch_in_chunks(batch["points"], self.ema_model, max_chunk_size=2**18)
        pred_img = pred_img[:img_size, :].reshape(self.img_data.shape).float().clamp(0.0, 1.0)
        pred_img = pred_img.cpu().numpy()

        img_path = f"{self.val_dir}/result.jpg"
        write_image(img_path, pred_img)


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = ImageSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'{hparams.output_dir}/{system.time}/ckpts/',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)

    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"{hparams.output_dir}/{system.time}/logs/",
                               name="",
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=5,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      gradient_clip_val=1.0,
                      strategy=None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    if hparams.val_only:
        trainer.predict(system, ckpt_path=hparams.ckpt_path)
        system.output_metrics(logger)
    else:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)
        trainer.predict()
        system.output_metrics(logger)