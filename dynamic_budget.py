from torchvision import models
from torchsummary import summary
from torch import nn
import torch
import random
import timeit

from models import get_model
import pytorch_lightning as pl
from torch.distributions import Bernoulli

# Implement training framework
class Dynamic_Budget_trainloop(pl.LightningModule):
  
  def __init__(self, dyn_bkn, agent, beg_cl_step=1):
    super().__init__()
    self.dyn_bkn = dyn_bkn
    self.agent = agent
    self.bound_alpha = 0.8
    self.penalty = -1.0
    self.beta = 1e-1
    self.cl_step = beg_cl_step
    self.test_out = []

    self.loss_fn = torch.nn.CrossEntropyLoss()
    self.loss_lst, self.top1_lst, self.top5_lst = [], [], []

  # key design : 
  #   1. min block usage
  #   2. wrong pred with penality 
  def get_reward(self, preds, labs, selected_blk):
    block_use = selected_blk.sum(1).float() / selected_blk.shape[1]
    sparse_reward = 1.0-block_use**2
    
    _, pred_idx = preds.max(1)
    matched = pred_idx==labs

    reward = sparse_reward
    reward[~matched] = self.penalty
    reward = reward.unsqueeze(1)
    
    # for mean acc, convert to float type
    return reward, matched.float()

  def pref_status(self, tag, policies, rewards, matched):
    perf_dict = {
      f'{tag}/accuracy' : matched.mean(),
      f'{tag}/reward' : rewards.mean(),
      f'{tag}/sparsity' : policies.sum(-1).mean(),
      f'{tag}/variance' : policies.sum(-1).std()
    }
    return perf_dict

  def training_step(self, batch, batch_index):
    loss_dict = {}
    im, lab, _ = batch

    # one-step MDP, classification env is simple..
    # predict selected block for each img, [bz, n_blk]
    select_pred = self.agent(im)

    # make baseline prediction with simple threshold (used in inference phase)
    base_select_pred = select_pred.clone()
    base_select_pred[base_select_pred<0.5] = 0.0
    base_select_pred[base_select_pred>=0.5] = 1.0

    # baseline prediction (Benoulli dist sampling 0/1)
    bound_prob = select_pred * self.bound_alpha + (1-select_pred) * (1-self.bound_alpha)
    distr = Bernoulli(bound_prob)
    select_pred = distr.sample()

    # release constraint by curriculum-training
    if self.cl_step < self.dyn_bkn.n_blk:
      base_select_pred[:, :-self.cl_step] = 1
      select_pred[:, :-self.cl_step] = 1
      policy_mask = torch.ones_like(select_pred)
      policy_mask[:, :-self.cl_step] = 0
    else:
      policy_mask = None

    # training bottleneck comes from go though self.dyn_bkn twice (timeit)
    base_pred = self.dyn_bkn(im, base_select_pred)
    base_reward, matched = self.get_reward(base_pred, lab, base_select_pred)
    
    pred = self.dyn_bkn(im, select_pred)
    reward, _ = self.get_reward(pred, lab, select_pred)
    
    # advantage reward measurement (Self-critical Sequence Training)
    advantage = reward - base_reward
    # Score function : https://pytorch.org/docs/stable/distributions.html#score-function
    #   we max reward == min rl_loss (by adding negative term)
    rl_loss = -1 * (advantage * distr.log_prob(select_pred))
    # mask for curriculum learning
    if policy_mask is not None:
      rl_loss = (policy_mask * rl_loss).sum()
    else:
      rl_loss = rl_loss.sum()
    loss_dict['train/rl_loss'] = rl_loss
    
    # maximum search space of drop block actions
    probs = bound_prob.clamp(1e-15, 1-1e-15)
    entropy_loss = ( -probs*torch.log(probs) ).sum()
    loss_dict['train/entropy_loss'] = entropy_loss

    total_loss = (rl_loss - (self.beta * entropy_loss)) / im.shape[0]
    loss_dict['train/total_loss'] = total_loss

    # since we use 'threshold' in inference phase, so we measure base_*
    perf_dict = self.pref_status('train', base_select_pred, base_reward, matched)
    loss_dict.update(perf_dict)
    self.log_dict(loss_dict, on_step=True, prog_bar=True)

    return total_loss

  def on_train_epoch_end(self):
    if self.cl_step < self.dyn_bkn.n_blk:
      self.cl_step += 1

  def validation_step(self, batch, batch_idx):
    im, lab, _ = batch

    # one-step MDP, classification env is simple..
    # drop-pred of each block, [bz, n_blk]
    base_select_pred = self.agent(im)

    # make base prediction with simple threshold
    base_select_pred[base_select_pred<0.5] = 0.0
    base_select_pred[base_select_pred>=0.5] = 1.0
    if self.cl_step < self.dyn_bkn.n_blk:
      base_select_pred[:, :-self.cl_step] = 1

    pred = self.dyn_bkn(im, base_select_pred)
    reward, matched = self.get_reward(pred, lab, base_select_pred)
    
    perf_dict = self.pref_status('valid', base_select_pred, reward, matched)
    self.log_dict(perf_dict, prog_bar=True)

  # called by Trainer.test(.), agnoistic with train_step..
  def test_step(self, batch, batch_idx):
    loss_dict = {}
    im, lab, _ = batch

    # one-step MDP, classification env is simple..
    # drop-pred of each block, [bz, n_blk]
    blk_pred = self.agent(im)

    # make best prediction with simple threshold
    drop_pred = blk_pred.clone()
    drop_pred[drop_pred<0.5] = 0.0
    drop_pred[drop_pred>=0.5] = 1.0

    pred = self.dyn_bkn(im, drop_pred, bin_path=True)
    g_flops = 0 * 1e-9

    _, pred_idx = pred.max(1)
    matched = (pred_idx==lab)
    acc = matched.float().mean()
    
    # if num of test data can't divided by batch size,
    #   this method will take weighted accuracy
    self.log('test_acc', acc, batch_size=1)
    print(acc)
    '''
    self.log_dict(
      {'test_acc': acc, 'Gflops': g_flops},
      on_step=True,
      on_epoch=True,
      prog_bar=True
    )
    '''

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3, verbose=True)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "train/total_loss",
            'interval': 'epoch',
            "frequency": 1
        },
    }
  