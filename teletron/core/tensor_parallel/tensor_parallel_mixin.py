from .layers import TeleColumnParallelLinear, TeleRowParallelLinear, TeleParallelRMSNorm, TeleParallelRMSNormBeta, TeleParallelRMSNormAlpha
from .mappings import reduce_mean, scatter_to_tensor_model_parallel_region

from megatron.core import mpu

import torch
import torch.nn as nn

class TensorParallelMixin:
    
    def __init__(self, config):
        self.config = config
        
    def enable_col_parallel(
        self, 
        linear_module: nn.Module,
        bias: bool=True,
        skip_bias_add: bool=False,
        gather_output: bool=False,
        skip_weight_param_allocation: bool=False,
        ):
        
        return TeleColumnParallelLinear(
                linear_module.in_features,
                linear_module.out_features,
                config=self.config,
                init_method=self.config.init_method,
                bias=bias,
                skip_bias_add=skip_bias_add,
                gather_output=gather_output,
                skip_weight_param_allocation=skip_weight_param_allocation,
                )  

            
    def enable_row_parallel(
        self, 
        linear_module: nn.Module,
        bias: bool=True,
        input_is_parallel: bool=True,
        skip_bias_add: bool=False,
        ):
        
        return TeleRowParallelLinear(
                linear_module.in_features,
                linear_module.out_features,
                config=self.config,
                init_method=self.config.init_method,
                bias=bias,
                input_is_parallel=input_is_parallel,
                skip_bias_add=skip_bias_add,
                )
        
    def enable_rms_norm_parallel(self, rmsnorm_module: nn.Module, dim):
        return TeleParallelRMSNorm( dim = dim, eps = rmsnorm_module.eps)
    
    def enable_rms_norm_parallel_beta(self, rmsnorm_module: nn.Module, dim):
        return TeleParallelRMSNormBeta( dim = dim, eps = rmsnorm_module.eps)
    
    def enable_attn_module_parallel(self, attn_module: nn.Module):
        from teletron.models.teleai.teleai_model import AttentionModule
        world_size = mpu.get_tensor_model_parallel_world_size()
        num_heads = attn_module.num_heads // world_size
        return AttentionModule(num_heads)
        
    
    def enable_self_attn_tensor_parallel(self, module: nn.Module):
        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size
        
        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.norm_query = self.enable_rms_norm_parallel(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel(module.norm_key, module.dim)
        
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        module.attn = self.enable_attn_module_parallel(module.attn)
        
    def enable_self_attn_tensor_parallel_beta(self, module: nn.Module):
        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size
        
        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.norm_query = self.enable_rms_norm_parallel_beta(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel_beta(module.norm_key, module.dim)
        
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        module.attn = self.enable_attn_module_parallel(module.attn)
        
        
    def enable_cross_attn_tensor_parallel(self, module: nn.Module):

        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size

        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        module.norm_query = self.enable_rms_norm_parallel(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel(module.norm_key, module.dim)
        
        if module.has_image_input:
            module.img_key = self.enable_col_parallel(module.img_key, gather_output=False) 
            module.img_value = self.enable_col_parallel(module.img_value, gather_output=False)
            module.norm_image_key = self.enable_rms_norm_parallel(module.norm_image_key, module.dim)
            
        module.attn = self.enable_attn_module_parallel(module.attn)
        module.attn2 = self.enable_attn_module_parallel(module.attn2)
        
    def enable_cross_attn_tensor_parallel_beta(self, module: nn.Module):

        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size

        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.norm_query = self.enable_rms_norm_parallel_beta(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel_beta(module.norm_key, module.dim)
        
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        
        if module.has_image_input:
            module.img_key = self.enable_col_parallel(module.img_key, gather_output=False) 
            module.img_value = self.enable_col_parallel(module.img_value, gather_output=False)
            module.norm_image_key = self.enable_rms_norm_parallel_beta(module.norm_image_key, module.dim)
            
        module.attn = self.enable_attn_module_parallel(module.attn)
        module.attn2 = self.enable_attn_module_parallel(module.attn2)
        
    def enable_ffn_tensor_parallel(self, ffn_module):
        ffn_module[0] = self.enable_col_parallel(ffn_module[0])
        ffn_module[2] = self.enable_row_parallel(ffn_module[2])
        
    def enable_rms_norm_parallel_alpha(self, rmsnorm_module: nn.Module, dim):
        return TeleParallelRMSNormAlpha( dim = dim, eps = rmsnorm_module.eps)
    
    def enable_cross_attn_tensor_parallel_alpha(self, module: nn.Module):

        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size

        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        module.norm_query = self.enable_rms_norm_parallel_alpha(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel_alpha(module.norm_key, module.dim)
        
        if module.has_image_input:
            module.img_key = self.enable_col_parallel(module.img_key, gather_output=False) 
            module.img_value = self.enable_col_parallel(module.img_value, gather_output=False)
            module.norm_image_key = self.enable_rms_norm_parallel_alpha(module.norm_image_key, module.dim)
            
        module.attn = self.enable_attn_module_parallel(module.attn)
        module.attn2 = self.enable_attn_module_parallel(module.attn2)
        
    def enable_self_attn_tensor_parallel_alpha(self, module: nn.Module):
        world_size = mpu.get_tensor_model_parallel_world_size()
        module.num_heads = module.num_heads // world_size
        
        module.query = self.enable_col_parallel(module.query, gather_output=False) 
        module.key = self.enable_col_parallel(module.key, gather_output=False)
        module.norm_query = self.enable_rms_norm_parallel_alpha(module.norm_query, module.dim)
        module.norm_key = self.enable_rms_norm_parallel_alpha(module.norm_key, module.dim)
        
        module.value = self.enable_col_parallel(module.value, gather_output=False)
        module.out_proj = self.enable_row_parallel(module.out_proj)
        module.attn = self.enable_attn_module_parallel(module.attn)