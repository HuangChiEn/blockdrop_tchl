from torch import nn
import torch
from pprint import pprint

import timm
from timm.models.vision_transformer import default_cfgs
from timm.models.vision_transformer import VisionTransformer

from torch_flops import TorchFLOPsByFX

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

    # freeze weight
    for param in self.base_model.parameters():
      param.requires_grad = False

    # we only extract the block-level content from vit, 
    #   for the finer layer, we haven't plan it yet..
    self.mod_blk_lst = self.base_model.blocks

    # currently, we only support shared_adaptive_layer
    #self.adaptive_layers = nn.Sequential(nn.Linear(768, 768), nn.ReLU())

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

  def forward(self, inp, select_indicator, bin_path=False, cal_flops=False):
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
      #blk_lst = [ blk for blk in self.mod_blk_lst ]
      #print(blk_lst)
      static_infer_path = nn.Sequential(*blk_lst)
      out = static_infer_path(inp.clone())
      if cal_flops:
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

