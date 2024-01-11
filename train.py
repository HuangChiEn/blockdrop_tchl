import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

from models import get_model, Dynamic_ViT_backbone, Agent_wrapper
from blockdrop import BlockDrop_net
from dataset import ImageNet_ds

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# add budget mechanism
def build_model_agent(model_type, agent_type):
  model = get_model(model_type, pretrained=True)
  dynamic_model = Dynamic_ViT_backbone(model.eval())

  agent = get_model(agent_type, pretrained=True, num_classes=dynamic_model.n_blk)
  agent = Agent_wrapper(agent)
  return dynamic_model, agent

def build_data_loader(tra_ds_root, val_ds_root, num_workers, batch_size, mod_cfg):
  img_ds = ImageNet_ds(tra_ds_root, val_ds_root)
  img_ds.train_transform = img_ds.test_transform = create_transform(**mod_cfg)
  
  img_ds.setup(stage='train')
  train_dataloader = img_ds.train_dataloader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

  img_ds.setup(stage='test', subset_ratio=0.2)
  val_dataloader = img_ds.test_dataloader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
  return train_dataloader, val_dataloader

def main(cfger):
  dyn_model, agent = build_model_agent(**cfger.model_params)
  mod_cfg = resolve_data_config(model=dyn_model)

  tra_ld, val_ld = build_data_loader(**cfger.data_params, mod_cfg=mod_cfg)
  
  # inspect the training-loop,
  # plz trace the forward(.) of BlockDrop_net
  blkdrop_setup = getattr(cfger, f"{cfger.exe_stage}_params")
  ba_dnn = BlockDrop_net(dyn_bkn=dyn_model, agent=agent, **blkdrop_setup)
  ba_dnn.exe_stage = cfger.exe_stage

  logger = WandbLogger(**cfger.logger['setup']) if cfger.logger['log_it'] \
                           else None 
  if logger: # log all hyper-params, not just for model..
    logger.log_hyperparams(vars(cfger))
  trainer = pl.Trainer(**cfger.trainer, logger=logger)
  trainer.fit(ba_dnn, tra_ld, val_ld)

def get_config():
  return '''
  # train or finetune
  exe_stage = 'train'

  [train_params]
    lr = 1e-3
    beta = 1e-1
    # for training only
    beg_cl_step = 1
    penalty = -1.0
    bound_alpha = 0.8

  [finetune_params]
    lr = 1e-4
    penalty = -5
    bound_alpha = 0.8
    
  [model_params]
    model_type = 'vit_base_patch8_224'
    agent_type = 'resnet18'
  
  [data_params]
    # root for colab setup!
    tra_ds_root = '/content/imagenette2/train'
    val_ds_root = '/content/imagenette2/val'
    num_workers = 2
    batch_size = 4 

  [trainer]
    accelerator = 'gpu'
    max_epochs = 12
    # debug for tra/val loop via fast testing!
    fast_dev_run = False
    log_every_n_steps = 1
  
  [logger]
    log_type = 'wandb'
    log_it = True
    [logger.setup]
      project = 'budget_dyn_nn'
      offline = True
  '''

if __name__ == "__main__":
  from easy_configer.Configer import Configer
  cfger = Configer(cmd_args=True)
  cfger.cfg_from_str( get_config() )

  main(cfger)
