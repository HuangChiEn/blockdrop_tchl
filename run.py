import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

from models import get_model, model_hub, Agent_wrapper
from blockdrop import BlockDrop_net
from dataset import ds_hub

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# add budget mechanism
def build_model_agent(model_tag, agent_tag):
  model = get_model(model_type, pretrained=True)
  tag = 'vit' if 'vit' in model_tag else 'resnet'
  dynamic_model = model_hub[tag](model)
  
  # source : https://github.com/Tushar-N/blockdrop/blob/ec52b36d38dc21335df539ac24e51462c6012b5c/models/resnet.py#L235
  # enable pretrained weight
  agent = get_model(agent_type, pretrained=True)
  agent = Agent_wrapper(agent, num_blocks=dynamic_model.n_blk)
  
  return dynamic_model, agent

def build_data_loader(dataset, tra_ds_root, tst_ds_root, val_sub_ratio, mod_cfg, **ds_args):
  ds_inst = ds_hub[dataset](tra_ds_root, tst_ds_root)
  if mod_cfg:
    img_ds.train_transform = img_ds.test_transform = create_transform(**mod_cfg)

  ds_inst.prepare_data()
  img_ds.setup(val_sub_ratio)

  train_dataloader = img_ds.train_dataloader(**ds_args)
  val_dataloader = img_ds.val_dataloader(**ds_args)
  test_dataloader = img_ds.test_dataloader(**ds_args)
  return train_dataloader, val_dataloader, test_dataloader


def main(cfger):
  # 1. get model & dataset
  dyn_model, agent = build_model_agent(**cfger.model_params)
  try:
    mod_cfg = resolve_data_config(model=dyn_model)
  except:
    mod_cfg = None
  tra_ld, val_ld, tst_ld = build_data_loader(**cfger.data_params, mod_cfg=mod_cfg)
  
  # 2. setup trainer utils
  lr_monitor = LearningRateMonitor(logging_interval='step')
  logger = WandbLogger(**cfger.logger['setup']) if cfger.logger['log_it'] \
                           else None 
  if logger: # log all hyper-params, not just for model..
    logger.log_hyperparams( vars(cfger) )
  
  # 3. dispatch to the corresponding procedure 
  # TODO : refactor here!
  trainer = pl.Trainer(**cfger.trainer, logger=logger, callbacks=[lr_monitor])
  if cfger.exe_stage == 'train':
    ba_dnn = BlockDrop_net(dyn_bkn=dyn_model, agent=agent, stage_flag=cfger.exe_stage, **cfger.train_params)
    trainer.fit(ba_dnn, tra_ld, val_ld)
  elif cfger.exe_stage == 'finetune':
    ba_dnn = BlockDrop_net.load_from_checkpoint(
      cfger.finetune_params['PATH'], dyn_bkn=dyn_model, agent=agent,
      stage_flag=cfger.exe_stage, **cfger.finetune_params 
    )
    trainer.fit(ba_dnn, tra_ld, val_ld)
  else:
    ba_dnn = BlockDrop_net.load_from_checkpoint(
      cfger.test_params['PATH'], dyn_bkn=dyn_model, agent=agent,
      stage_flag=cfger.exe_stage, **cfger.train_params # dummy input..
    )
    ba_dnn.setup_test_args(**cfger.test_args)
    tst_ld = val_ld # valid set is just test set with more sample
    trainer.test(ba_dnn, dataloaders=tst_ld)

if __name__ == "__main__":
  import os
  from easy_configer.Configer import Configer
  cfg_path = os.environ['RUN_CFG'] if os.environ['RUN_CFG'] else './config.ini'
  cfger = Configer(cmd_args=True)
  cfger.cfg_from_ini(cfg_path)

  main(cfger)
