from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import timm
from timm.models.vision_transformer import default_cfgs
from timm.models.vision_transformer import VisionTransformer
from timm.models.resnet import ResNet

from torch_flops import TorchFLOPsByFX
from pprint import pprint

def get_model(model_type, pretrained=True, view_cfg=False, **kwargs):
  if view_cfg:
    pprint(default_cfgs[model_type].default_with_tag)
  else:
    model = timm.create_model(model_type, pretrained=pretrained, **kwargs)
    return model

# Implement block pool and dynamic routing
class Dynamic_ViT_backbone(nn.Module):
  
  def __init__(self, base_model):
    def arg_chk(base_model):
      if not isinstance(base_model, VisionTransformer):
        raise RuntimeError("Only allow pure timm vanilla-ViT model")
      
    super().__init__()

    arg_chk(base_model)
    self.base_model = base_model

    # we only extract the block-level content from vit, 
    #   for the finer layer, we haven't plan it yet..
    self.mod_blk_lst = self.base_model.blocks

  def freeze(self):
    # freeze weight
    for param in self.base_model.parameters():
      param.requires_grad = False
    # update module block list for sure 
    self.mod_blk_lst = self.base_model.blocks

  def non_freeze(self):
    # freeze weight
    for param in self.base_model.parameters():
      param.requires_grad = True
    # update module block list for sure 
    self.mod_blk_lst = self.base_model.blocks

  @property
  def n_blk(self):
    return len(self.mod_blk_lst)

  ## We discompose the timm vit forward method,
  ##  and make self.block layers could be dynamic routing. (keep the other arch same)
  # https://github.com/huggingface/pytorch-image-models/blob/b5a4fa9c3be6ac732807db7e87d176f5e4fc06f1/timm/models/vision_transformer.py#L668C5-L695C1
  def _forward_features(self, x):
    x = self.base_model.patch_embed(x)
    x = self.base_model._pos_embed(x)
    x = self.base_model.patch_drop(x)
    x = self.base_model.norm_pre(x)
    return x
  # https://github.com/huggingface/pytorch-image-models/blob/b5a4fa9c3be6ac732807db7e87d176f5e4fc06f1/timm/models/vision_transformer.py#L680C8-L680C8
  def _forward_head(self, x, pre_logits = False):
    # we move this line from _forward_features to here..
    x = self.base_model.norm(x)
    
    if self.base_model.attn_pool is not None:
        x = self.base_model.attn_pool(x)
    elif self.base_model.global_pool == 'avg':
        x = x[:, self.base_model.num_prefix_tokens:].mean(dim=1)
    elif self.base_model.global_pool:
        x = x[:, 0]  # class token
    x = self.base_model.fc_norm(x)
    x = self.base_model.head_drop(x)
    return x if pre_logits else self.base_model.head(x)

  def forward(self, inp, select_indicator, bin_path=False, cal_flops=False, benchmark=False):
    # training phase still "go through all blocks"
    if not bin_path:
      inp = self._forward_features(inp)
      for blk_id, block in enumerate(self.mod_blk_lst):
        prev_inp = inp # record inp before feed into block
        out = block(inp)
        msk = torch.zeros_like(out) # default, all feat taken from previous block
        
        select_blk_idx = select_indicator[:, blk_id].nonzero()
        msk[select_blk_idx] = 1.0 
        inp = out * msk + prev_inp * (1-msk)
        
      return self._forward_head(inp)

    # inference phase enable binary path (drop some blocks)
    else:
      if inp.shape[0] != 1:
        raise RuntimeError("The binarized inference path suppose 'batch size = 1'!!")
      
      inp = self._forward_features(inp)

      # construct static binary inference path
      blk_lst = []
      blk_idxs = (select_indicator[0, :]!=0).nonzero()
      for blk_id in blk_idxs:
        block = self.mod_blk_lst[blk_id]
        blk_lst.append(block)

      static_infer_path = nn.Sequential(*blk_lst) if not benchmark \
                              else nn.Sequential(self.mod_blk_lst)
      out = static_infer_path(inp.clone())
      if cal_flops or benchmark:
        benchmrk_mod = TorchFLOPsByFX(static_infer_path)
        benchmrk_mod.propagate(inp)
        flops = benchmrk_mod.print_total_flops(show=False)
        return self._forward_head(out), flops
      
      return self._forward_head(out)


# Implement layer pool and dynamic routing
class Dynamic_Resnet_backbone(nn.Module):
  
  def __init__(self, base_model, resnet_cfg=[3, 4, 23, 3]):
    def arg_chk(base_model):
      if not isinstance(base_model, ResNet):
        raise RuntimeError("Only allow pure timm vanilla-ViT model")
    
    def create_blk_lst(model):
      mod_blk_lst = []
      named_dict = { nam : net for nam, net in model.named_modules() }
      for lay_name in self.id2lay.values():
        if lay_name not in named_dict.keys():
          raise RuntimeError(f"Missing pretrained model layer {lay_name}")
        mod_blk_lst.append( named_dict[lay_name] )
      return mod_blk_lst

    super().__init__()

    arg_chk(base_model)
    self.base_model = base_model

    # layer cfg record num of sub-layer for each layer in resnet 
    self.id2lay = { idx:f'layer1.{idx}' for idx in range(0, resnet_cfg[0]) }
    for cfg_id, lay_rng in enumerate(resnet_cfg[1:], 0):
      self.id2lay.update( { resnet_cfg[cfg_id]+idx:f'layer2.{idx}' for idx in range(0, lay_rng) } )
      cnt += resnet_cfg[cfg_id]

    # we only extract the layer-level content from resnet
    #  note :  self.mod_blk_lst share same id with self.base_model!
    self.mod_blk_lst = create_blk_lst(self.base_model)

  def freeze(self):
    # freeze weight
    for layer_blk in self.mod_blk_lst:
      for lay_param in layer_blk.parameters():
        lay_param.requires_grad = False

  @property
  def n_blk(self):
    return len(self.mod_blk_lst)

  ## We discompose the timm resnet forward method,
  ##  and keep the other arch same
  # https://github.com/huggingface/pytorch-image-models/blob/8c663c4b8607981eca31b836392ef67f68fcd12f/timm/models/resnet.py#L556
  def _forward_features(self, x):
    x = self.base_model.conv1(x)
    x = self.base_model.bn1(x)
    x = self.base_model.act1(x)
    x = self.base_model.maxpool(x)
    return x
  # https://github.com/huggingface/pytorch-image-models/blob/8c663c4b8607981eca31b836392ef67f68fcd12f/timm/models/resnet.py#L571
  def _forward_head(self, x, pre_logits = False):
    x = self.base_model.global_pool(x)
    if self.base_model.drop_rate:
        x = F.dropout(x, p=float(self.base_model.drop_rate), training=self.base_model.training)
    return x if pre_logits else self.base_model.fc(x)

  def forward(self, inp, select_indicator, bin_path=False, cal_flops=False, benchmark=False):
    # training phase still "go through all blocks"
    if not bin_path:
      inp = self._forward_features(inp)
      for blk_id, block in enumerate(self.mod_blk_lst):
        prev_inp = inp # record inp before feed into block
        out = block(inp)
        msk = torch.zeros_like(out) # default, all feat taken from previous block
        
        select_blk_idx = select_indicator[:, blk_id].nonzero()
        msk[select_blk_idx] = 1.0 
        inp = out * msk + prev_inp * (1-msk)
        
      return self._forward_head(inp)

    # inference phase enable binary path (drop some blocks)
    else:
      if inp.shape[0] != 1:
        raise RuntimeError("The binarized inference path suppose 'batch size = 1'!!")
      
      inp = self._forward_features(inp)

      # construct static binary inference path
      blk_lst = []
      blk_idxs = (select_indicator[0, :]!=0).nonzero()
      for blk_id in blk_idxs:
        block = self.mod_blk_lst[blk_id]
        blk_lst.append(block)
      
      static_infer_path = nn.Sequential(*blk_lst) if not benchmark \
                              else nn.Sequential(self.mod_blk_lst)
      out = static_infer_path(inp.clone())
      if cal_flops or benchmark:
        benchmrk_mod = TorchFLOPsByFX(static_infer_path)
        benchmrk_mod.propagate(inp)
        flops = benchmrk_mod.print_total_flops(show=False)
        return self._forward_head(out), flops
      
      return self._forward_head(out)

class Agent_wrapper(nn.Module):
  
  def __init__(self, base_model):
    super().__init__()
    self.base_model = base_model
    
  def forward(self, inp):
    # load pretrain model with n_cls=n_blk
    #   let it output correct shape 
    out_logit = self.base_model(inp) 
    # turn logit to prob
    return torch.sigmoid(out_logit)


# For testing the correctness of dynamic ViT
class Static_ViT_backbone(pl.LightningModule):
  def __init__(self):  
    super().__init__()
    self.base_model = get_model('vit_base_patch16_224', pretrained=True).eval()

  def test_step(self, batch, batch_idx):
    im, lab, _ = batch
    pred = self.base_model(im)
    _, pred_idx = pred.max(1)
    acc = (pred_idx==lab).float()
    self.log_dict({'test_acc': acc}, on_step=True, on_epoch=True, prog_bar=True)

class Static_ResNet_backbone(pl.LightningModule):
  def __init__(self):  
    super().__init__()
    self.base_model = get_model('resnet101', pretrained=True).eval()

  def test_step(self, batch, batch_idx):
    im, lab, _ = batch
    pred = self.base_model(im)
    _, pred_idx = pred.max(1)
    acc = (pred_idx==lab).float()
    self.log_dict({'test_acc': acc}, on_step=True, on_epoch=True, prog_bar=True)
    
# yeah, it just a py file, not package,
# so we need to define hub at the end to read all classes..
model_hub = {
  'vit' : Dynamic_ViT_backbone,
  'resnet' : Dynamic_Resnet_backbone,
  'static_vit' : Static_ViT_backbone,
  'static_resnet' : Static_ResNet_backbone
}
