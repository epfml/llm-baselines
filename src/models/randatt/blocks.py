import torch 
import torch.nn as nn 
import math 

from torch.autograd import Variable

#import src.models.randatt.attention
from .attention import SelfAttention, CrossAttention, RandomBlockSelfAttention, RandomBlockCrossAttention
from .tools import causal_mask, alibi_shift

from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim = (256,), act=nn.GELU(), bias: bool = False):

        super().__init__()
        
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 
        self.act = act 

        self.model = nn.Sequential()
        hidden_dim = (input_dim,) + hidden_dim 

        for i in range(len(hidden_dim) - 1):

            self.model.append(nn.Linear(hidden_dim[i], hidden_dim[i+1], bias=bias))
            self.model.append(act)
        
        self.model.append(nn.Linear(hidden_dim[-1], output_dim, bias=bias))
    
    def forward(self, x):

        return self.model(x)




class EncoderBlock(nn.Module):

    def __init__(self, model_dim, n_heads=2, dropout_rate=0.1, act=nn.GELU(), block_dim=None, bias: bool= False):

        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate 
        self.act = act 
        self.bias= bias
        self.attention = SelfAttention(model_dim, n_heads, act, bias= bias)
        self.mlp = MLP(model_dim, model_dim, 1 * (4*model_dim, ), bias = bias)

        self.norm1 = LayerNorm(model_dim, bias = bias)
        self.norm2 = LayerNorm(model_dim, bias = bias)

        self.drop_att = nn.Dropout(dropout_rate)
        self.drop_mlp = nn.Dropout(dropout_rate)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #     elif isinstance(module, LayerNorm):  # Handle LayerNorm explicitly
    #         torch.nn.init.ones_(module.weight)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)

    def attention_fn(self, x, mask=None, shift=None):
        att, keys, values = self.attention(self.norm1(x), mask=mask, shift=shift)
        att = self.drop_att(att)
        #x = self.norm1(x + att)
        x = x + att
        return x, keys, values

    def mlp_fn(self, x):    
        
        x_mlp = self.mlp(self.norm2(x))
        x_mlp = self.drop_mlp(x_mlp)
        #x = self.norm2(x + x_mlp)
        x = x + x_mlp
        return x

    def forward(self, x, mask=None, shift=None):

        device = x.device  # Ensure new tensors align with `x`

        # Update tensors to use the same device
        if mask is not None:
            mask = mask.to(device)
        if shift is not None:
            shift = shift.to(device)


        x, keys, values = self.attention_fn(x, mask, shift)
        x = self.mlp_fn(x)
        
        return x#, keys, values 


class DecoderBlock(EncoderBlock):

    def __init__(self, model_dim, n_heads, dropout_rate=0.1, act=nn.GELU()):

        super().__init__(model_dim, n_heads, dropout_rate, act)

        self.cross_attention = CrossAttention(model_dim, n_heads, act)
        self.cross_norm = nn.LayerNorm(model_dim)
        self.cross_drop = nn.Dropout(dropout_rate)

    def cross_attention_fn(self, x, k, v, shift):

        x_cross = self.cross_attention(x, k, v, shift=shift)
        x_cross = self.cross_drop(x_cross)
        x = self.cross_norm(x + x_cross)

        return x

    def forward(self, x, k, v, mask=None, shift=None):
        
        x, _, __ = self.attention_fn(x, mask=mask, shift=shift)
        x = self.cross_attention_fn(x, k, v, shift=shift)
        x = self.mlp_fn(x)

        return x 


class LightEncoderBlock(EncoderBlock):

    def __init__(self, model_dim, block_dim=100, n_heads=2, dropout_rate=0.1, act=nn.GELU(),bias: bool = False):
        print(f"Initializing LightEncoderBlock with block_dim={block_dim}")
        super().__init__(model_dim, n_heads, dropout_rate, act, bias= bias )
        self.attention = RandomBlockSelfAttention(model_dim, block_dim, n_heads, act, bias= bias)


class LightDecoderBlock(DecoderBlock):

    def __init__(self, model_dim, block_dim=100, n_heads=2, dropout_rate=0.1, act=nn.GELU()):

        super().__init__(model_dim, n_heads, dropout_rate, act)
        self.attention = RandomBlockSelfAttention(model_dim, block_dim, n_heads, act)
        self.cross_attention = RandomBlockCrossAttention(model_dim, block_dim, n_heads, act)


def test():
    x = Variable(torch.randn(5, 15, 32), requires_grad=True)

    enc = EncoderBlock(32, 2)

    o, k, v = enc(x)

    print("Encoder o, k, v")
    print(o.shape, k.shape, v.shape)

    lenc = LightEncoderBlock(32, 8, 2)
    
    mask = causal_mask(x, 4)
    shift = alibi_shift(x)

    print("Mask and shift")
    print(mask.shape, shift.shape)

    o, k, v = lenc(x, mask=mask, shift=shift)

    print("Light encoder o, k, v")
    print(o.shape, k.shape, v.shape)

    dec = LightDecoderBlock(32, 8, 2)

    o = dec(o, k, v)

    print("Decoder o")
    print(o.shape)

if __name__ == '__main__':
    test()