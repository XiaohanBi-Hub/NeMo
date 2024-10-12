"""
Copy from Huggingface
Then adding some codes for modular training
"""
import torch
import torch.nn as nn
import math

from transformers import DeiTForImageClassification
from transformers.models.deit.configuration_deit import DeiTConfig
from transformers.models.deit.modeling_deit import DeiTSelfAttention, DeiTLayer, DeiTAttention, DeiTIntermediate, DeiTOutput
from typing import Optional, Tuple, Union
import copy

def calculate_param(model, num=0):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Linear):
            num += child.in_features * child.out_features + child.out_features
        else:
            num = calculate_param(child, num)
    return num

class DeiTSelfAttentionMask(nn.Module):
    # self-attention with mask and padding
    # mask could remove part of the linear weights and reduce calculating costs
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.q_padding_in, self.k_padding_in, self.v_padding_in = None,None,None
        self.q_padding_out, self.k_padding_out, self.v_padding_out = None,None,None

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def qkv_padding(self, qkv_layer, padding):
        paded_output = torch.zeros(self.batch_size, self.patch_size, self.all_head_size).to(qkv_layer.device)
        paded_output[:,:,padding != 0] = qkv_layer
        return paded_output

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        self.batch_size = hidden_states.shape[0]
        self.patch_size = hidden_states.shape[1]

        # Padding required!
        query_layer = self.transpose_for_scores(self.qkv_padding(mixed_query_layer,self.q_padding_out))
        key_layer = self.transpose_for_scores(self.qkv_padding(mixed_key_layer,self.k_padding_out))
        value_layer = self.transpose_for_scores(self.qkv_padding(mixed_value_layer,self.v_padding_out))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # Remove 0 parts
        context_layer = context_layer[:,:,self.v_padding_out!=0]

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

def get_deit_module(mt_model, module_mask, deit_config, mask_attention_list, i = 0, att_i = 0, no_mask = 0):
    # Three steps to construct the module 
    # 1. replace linear to remove zero neurals
    # 2. search qkv and replace each linear into a new self-attention class
    # 3. replace self-attention part in the original model
    # no_mask indicates the top x encoders should not be masked
    for name, child in mt_model.named_children():
        # print(name)
        if name in ['layernorm']:
            setattr(mt_model, name, nn.LayerNorm(((module_mask[-1]!=0).int().sum(),), eps=deit_config.layer_norm_eps))
        if isinstance(child, torch.nn.Linear): 
            if no_mask>2:
                no_mask-=1
                continue
            
            if name in ['classifier'] or "classifier" in name:
                break
            original_weight = child.weight.data
            original_bias = child.bias.data
            if name in ['query']:
                # for qkv, 
                index_in = (module_mask[i-1]> 0).int() if i>0 else torch.ones_like(module_mask[0])
                index_out = torch.einsum('i,i->i',module_mask[i],(module_mask[i+1] > 0).int())

                new_q = torch.nn.Linear(index_in.sum().item(), index_out.int().sum().item())
                new_q.weight.data = torch.einsum('ij,i->ij',original_weight[index_out != 0, :][:, index_in != 0], index_out[index_out != 0])
                new_q.bias.data = original_bias[index_out != 0]
                
                setattr(mask_attention_list[att_i], 'query', new_q)
                mask_attention_list[att_i].q_padding_in = index_in
                mask_attention_list[att_i].q_padding_out = index_out

            elif name in ['key']:
                index_in = (module_mask[i-2]> 0).int() if i>1 else torch.ones_like(module_mask[1])
                index_out = torch.einsum('i,i->i', (module_mask[i-1]> 0).int() ,module_mask[i])

                new_k = torch.nn.Linear(index_in.sum().item(), (index_out>0).int().sum().item())
                new_k.weight.data = torch.einsum('ij,i->ij',original_weight[index_out != 0, :][:, index_in != 0], index_out[index_out != 0])
                new_k.bias.data = original_bias[index_out != 0]

                setattr(mask_attention_list[att_i], 'key', new_k)
                mask_attention_list[att_i].k_padding_in = index_in
                mask_attention_list[att_i].k_padding_out = index_out

            elif name in ['value']:
                index_in = (module_mask[i-3]> 0).int() if i>2 else torch.ones_like(module_mask[2])
                index_out = module_mask[i]

                new_v = torch.nn.Linear(index_in.sum().item(), (index_out>0).int().sum().item())
                new_v.weight.data = torch.einsum('ij,i->ij',original_weight[index_out != 0, :][:, index_in != 0], index_out[index_out != 0])
                new_v.bias.data = original_bias[index_out != 0]

                setattr(mask_attention_list[att_i], 'value', new_v)
                mask_attention_list[att_i].v_padding_in = index_in
                mask_attention_list[att_i].v_padding_out = index_out

                att_i += 1
            else:
                index_in = (module_mask[i-1] > 0).int()
                index_out = module_mask[i]
                
                new_layer = torch.nn.Linear(index_in.sum().item(), (index_out>0).int().sum().item())
                new_layer.weight.data = original_weight[index_out != 0, :][:, index_in != 0]
                new_layer.bias.data = original_bias[index_out != 0]
                setattr(mt_model, name, new_layer)
            i += 1
        elif isinstance(child, nn.LayerNorm):
            if i > len(module_mask):
                break
            if no_mask>0:
                no_mask-=1
                continue
            if name in ['layernorm_before']:
                index_in = (module_mask[i-7]> 0).int() if i>6 else torch.ones_like(module_mask[2])
                layernorm_before = nn.LayerNorm((index_in.sum(),), eps=deit_config.layer_norm_eps)
                setattr(mt_model, name, layernorm_before)
            elif name in ['layernorm_after']:
                index_out = (module_mask[i-3]> 0).int()
                layernorm_after = nn.LayerNorm((index_out.sum(),), eps=deit_config.layer_norm_eps)
                setattr(mt_model, name, layernorm_after)
        else:
            mask_attention_list, i, att_i, no_mask = get_deit_module(child, module_mask, deit_config, mask_attention_list, i, att_i, no_mask)
    return mask_attention_list, i, att_i, no_mask

def replace_deitatt(mt_model, mask_attention_list, no_mask=0, att_i=0):
    for name, child in mt_model.named_children():
        if isinstance(child, DeiTSelfAttention):
            if no_mask>0:
                no_mask-=1
                continue
            setattr(mt_model, name, mask_attention_list[att_i])
            att_i+=1
        else:
            att_i, no_mask= replace_deitatt(child, mask_attention_list, no_mask, att_i)
    return att_i, no_mask

def replace_deitlayer(mt_model, each_layer_mask, config, no_mask=0, count=0):
    for name, child in mt_model.named_children():
        if isinstance(child, DeiTLayer):
            if no_mask>0:
                no_mask-=1
                continue
            class DeiTLayerMask(DeiTLayer):
                def __init__(self, config: DeiTConfig, layer_mask) -> None:
                    super().__init__(config)
                    self.attention = child.attention
                    self.intermediate = child.intermediate
                    self.output = child.output
                    self.layernorm_before = child.layernorm_before
                    self.layernorm_after = child.layernorm_after
                    self.layer_mask = layer_mask
                    self.config = config
                def forward(
                    self,
                    hidden_states: torch.Tensor,
                    head_mask: Optional[torch.Tensor] = None,
                    output_attentions: bool = False,
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
                    self_attention_outputs = self.attention(self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                                                            head_mask,
                                                            output_attentions=output_attentions,)
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

                    # first residual connection
                    # pad it to b*p*hidden_size
                    tmp1 = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], self.config.hidden_size, device=hidden_states.device)
                    # print('attention_output.shape',attention_output.shape)
                    tmp1[:,:,self.layer_mask[3]!=0] += attention_output
                    if len(self.layer_mask)==6:
                        # First encoder
                        tmp1 += hidden_states
                    else:
                        tmp1[:,:,self.layer_mask[6]!=0] += hidden_states
                    tmp2 = tmp1
                    hidden_states = tmp1[:,:,self.layer_mask[3]!=0]
                    # in ViT, layernorm is also applied after self-attention
                    layer_output = self.layernorm_after(hidden_states)
                    layer_output = self.intermediate(layer_output)

                    # second residual connection is done here
                    layer_output = self.output(layer_output, tmp2[:,:,self.layer_mask[5]!=0])

                    outputs = (layer_output,) + outputs

                    return outputs
            # Input the 6 linear mask and previous Encoder mask for residual; Put it into the last one
            layer_mask = each_layer_mask[6*count:6*(count+1)] 
            if count!=0:
                layer_mask.append(each_layer_mask[6*count-1])
            count += 1
            setattr(mt_model, name, DeiTLayerMask(config, layer_mask))
        else:
            replace_deitlayer(child, each_layer_mask, config, no_mask, count)

def deit_module(model_param, module_mask, target_classes):
    pic_size=32
    deit_config = DeiTConfig(
        num_labels=10,
        hidden_size=384,
        intermediate_size=1536,
        image_size=pic_size,
        patch_size=4,
    )
    mt_model = DeiTForImageClassification(deit_config)
    # replace_linear_deit(mt_model, replace_attention=True)
    # Only load model parameters, not mask generator
    mt_model.load_state_dict(model_param, strict=False)
    mt_model = copy.deepcopy(mt_model)

    hs = deit_config.hidden_size
    linear_size_each_encoder = [hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,
                                hs, hs, hs, hs, 4*hs, hs,]

    assert len(module_mask) == sum(linear_size_each_encoder)
    no_mask=1
    each_layer_mask = []
    start = 0
    for size in linear_size_each_encoder:
        end = start + size
        each_layer_mask.append(module_mask[start:end])
        start = end
    mask_attention_list = [ DeiTSelfAttentionMask(deit_config) for j in range(12) ]
    # 6 for linear and 2 for layernorm
    mask_attention_list, _, _, _= get_deit_module(mt_model, each_layer_mask, deit_config, mask_attention_list, no_mask = 6*no_mask+2)
    _, _ = replace_deitatt(mt_model, mask_attention_list, no_mask)
    replace_deitlayer(mt_model, each_layer_mask, deit_config, no_mask)

    out_feature_encoder = (each_layer_mask[-1]!=0).int().sum()
    # mt_model.cls_classifier = torch.nn.Sequential(
    #         torch.nn.ReLU(True),
    #         torch.nn.Linear(out_feature_encoder, len(target_classes)),
    #     )
    # mt_model.distillation_classifier = torch.nn.Sequential(
    #         torch.nn.ReLU(True),
    #         torch.nn.Linear(out_feature_encoder, len(target_classes)),
    #     )
    mt_model.classifier = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Linear(out_feature_encoder, len(target_classes)),
        )

    return mt_model


if __name__ == "__main__":
    DEVICE = torch.device(f"cuda:0")
    model_name = 'deit_s'
    mt_model_save_path = '/home/bixh/Documents/NeMo/data/data/modular_trained/deit_s_cifar10/lr_0.05_0.05_a0.1_t0.2_bs_128.pth'
    mt_model_param = torch.load(mt_model_save_path, map_location="cuda:0")
    # load modules' masks and the pretrained model
    modules_masks_save_path = f'/home/bixh/Documents/NeMo/data/data/modular_trained/deit_s_cifar10/lr_0.05_0.05_a0.1_t0.2_bs_128/mask_thres_0.9.pth'
    all_modules_masks = torch.load(modules_masks_save_path, map_location="cuda:0")
    target_classes=[0,1,2,3,4,5,6,7,8]

    def generate_target_module(target_classes, module_mask_path):
        # load modules' masks and the pretrained model
        all_modules_masks = torch.load(module_mask_path, map_location=DEVICE)
        target_module_mask = (torch.sum(all_modules_masks[target_classes], dim=0) > 0).int()
        # generate modules_arch by removing kernels from the model.
        mt_model_param = torch.load(mt_model_save_path, map_location=DEVICE)
        kernel_rate = torch.sum(target_module_mask) / len(target_module_mask)
        print(f'Kernel Rate: {kernel_rate:.2%}')
    generate_target_module(target_classes, modules_masks_save_path)
