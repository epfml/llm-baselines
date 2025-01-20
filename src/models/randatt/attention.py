import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

from typing import Callable

#import src.models.randatt.tools
from .tools import *

#
# Attention operation for a ready q, k, v tuple
#
def attention(q, k, v, mask=None, shift=None):

    d = q.shape[-1]

    #
    # Compute the attention logits - Q of shape (*, N, D) and K of shape (*, N, D) are mapped to Att of shape (*, N, N)
    #
    log_att = q @ k.transpose(-1, -2) / math.sqrt(d)
    
    #
    # Apply the mask by assigning ~= -inf to masked logits
    # 
    log_att = log_att if mask is None else log_att * mask - 1e6 * (1-mask)

    #
    # Apply the shift (e.g., ALiBi) to the logits
    #
    log_att = log_att if shift is None else log_att - shift
    
    #
    # Obtain attention weights from the logits
    #
    att = F.softmax(log_att, dim=-1)

    #
    # Compute the attention output by mixing v with attention weights
    #
    return att @ v  




class SelfAttention(nn.Module):

    def __init__(self, model_dim: int, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__()

        self.model_dim = model_dim
        self.n_heads = n_heads
        self.act = act 

        self.lin = nn.Linear(model_dim, 3 * model_dim)
        self.lin_out = nn.Identity() if n_heads == 1 else nn.Linear(model_dim, model_dim)


    def forward(self, x, mask=None, shift=None):

        #
        # Compute q, k, v from x at once; shape (*, N, 3 x D)
        #
        qkv = self.lin(x)

        #
        # Chunk qkv into different heads; shape (H, *, N, 3 x D / H)
        #
        qkv = torch.stack(
            qkv.chunk(self.n_heads, dim=-1)
        )

        #
        # Duplicate mask and shift for each head
        #
        if mask is not None: 
            mask = mask.unsqueeze(0).repeat((self.n_heads,) + len(mask.shape) * (1,))

        if shift is not None:
            shift = shift.unsqueeze(0).repeat((self.n_heads,) + len(shift.shape) * (1,))
            scale = torch.exp(-math.log(2) * torch.arange(self.n_heads) / 2).reshape((self.n_heads,) + len(shift.shape[:-1]) * (1,))
            shift = scale * shift

        #
        # Extract q, k, v
        #
        q, k, v = qkv.chunk(3, dim=-1)

        #
        # Compute self-attention
        #
        o = attention(q, k, v, mask, shift)

        #
        # Combine outputs from all heads
        #
        o = o.chunk(self.n_heads, dim=0)
        o = torch.cat(o, dim=-1).squeeze(0)
        o = self.lin_out(o)

        k = k.chunk(self.n_heads, dim=0)
        k = torch.cat(k, dim=-1).squeeze(0)

        v = v.chunk(self.n_heads, dim=0)
        v = torch.cat(v, dim=-1).squeeze(0)

        #
        # Return the output and the k, v pair for the decoder to use
        #
        return o, k, v




class RandomBlockSelfAttention(SelfAttention):

    def __init__(self, model_dim: int, block_dim: int = 100, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__(model_dim, n_heads, act)

        self.block_dim = block_dim

    def forward(self, x, mask=None, shift=None):

        #
        # Get the input sequence size and the number of blocks
        # 
        device = x.device

        seq_size= x.shape[-2]
        n_blocks = math.ceil(seq_size / self.block_dim)

        #
        # Pad the sequence so that the size can be evenly split among blocks
        #
        seq_full = n_blocks * self.block_dim
        pad_dims = 3 * (0,) + (seq_full - seq_size,)
        x_pad = F.pad(x, pad_dims).to(device)
   
        if mask is not None:
            mask = F.pad(mask, (0, seq_full - seq_size)).to(device)
        if shift is not None:
            shift = F.pad(shift, (0, seq_full - seq_size)).to(device)

        #
        # Pad also the mask and the shift if they are meant to be used
        #
        pad_dims = 2 * (0, seq_full - seq_size)
        mask_pad = None if mask is None else F.pad(mask, pad_dims)
        shift_pad = None if shift is None else F.pad(shift, pad_dims)

        #
        # Get random permutations of the sequence indexes
        #
        randvals = torch.rand(x_pad.shape[:-1],device=device)
        rand_indexes = torch.argsort(randvals, dim=-1)

        #
        # Align them with the modeled sequence and it split across heads
        #
        indexes_model = rand_indexes.unsqueeze(-1).repeat(len(x.shape[:-1]) * (1,) + (self.model_dim,)).to(device)
        indexes_heads = torch.stack(
            torch.chunk(indexes_model, self.n_heads, dim=-1)
        )

        #
        # Permute inputs to attention
        #
        x_rand = permute_sequence(x_pad, rand_indexes)
        mask_rand = None if mask is None else permute_matrix(mask_pad, rand_indexes, rand_indexes)
        shift_rand = None if shift is None else permute_matrix(shift_pad, rand_indexes, rand_indexes)

        #
        # Split the inputs into blocks and stack them along the first dimension
        #
        x_rand = torch.stack(
            torch.chunk(x_rand, n_blocks, dim=-2)
        )
        mask_rand = None if mask is None else extract_diagonal_blocks(mask_rand, n_blocks, self.block_dim)
        shift_rand = None if shift is None else extract_diagonal_blocks(shift_rand, n_blocks, self.block_dim)
                
        #
        # Compute self-attention for the permuted inputs
        #
        o_rand, k_rand, v_rand = super(RandomBlockSelfAttention, self).forward(x_rand, mask_rand, shift_rand)

        #
        # Reassemble the outputs
        #
        o_rand = torch.cat(
            torch.chunk(o_rand, n_blocks, 0), dim=-2
        ).squeeze(0)
        k_rand = torch.cat(
            torch.chunk(k_rand, n_blocks, 0), dim=-2
        ).squeeze(0)
        v_rand = torch.cat(
            torch.chunk(v_rand, n_blocks, 0), dim=-2
        ).squeeze(0)

        #
        # Redistribute the outputs to the right indexes
        # 
        o = torch.scatter(o_rand.clone(), -2, indexes_model, o_rand)
        k = torch.scatter(k_rand.clone(), -2, indexes_model, k_rand)
        v = torch.scatter(v_rand.clone(), -2, indexes_model, v_rand)

        #
        # Cut off the padding
        #
        o = o[..., :seq_size, :]
        k = k[..., :seq_size, :]
        v = v[..., :seq_size, :]

        return o.to(device), k.to(device), v.to(device)



class CrossAttention(SelfAttention):

    def __init__(self, model_dim: int, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__(model_dim, n_heads, act)
        
        self.lin = nn.Linear(model_dim, model_dim)

    def forward(self, x, k, v, mask=None, shift=None):

        #
        # Compute q from x 
        #
        q = self.lin(x)

        #
        # Chunk inputs into different heads
        #
        q = torch.stack(
            q.chunk(self.n_heads, dim=-1)
        )
        k = torch.stack(
            k.chunk(self.n_heads, dim=-1)
        )
        v = torch.stack(
            v.chunk(self.n_heads, dim=-1)
        )

        #
        # Compute cross-attention
        #
        o = attention(q, k, v, mask, shift)

        #
        # Combine outputs from all heads
        #
        o = o.chunk(self.n_heads, dim=0)
        o = torch.cat(o, dim=-1).squeeze(0)
        o = self.lin_out(o)

        return o



class RandomBlockCrossAttention(CrossAttention):

    def __init__(self, model_dim: int, block_dim: int = 100, n_heads: int = 2, act: Callable = nn.GELU()):

        super().__init__(model_dim, n_heads, act)

        self.block_dim = block_dim
    
    def forward(self, x, k, v, mask=None, shift=None):
        #
        # Define the number of blocks from the keys/values since they are usually longer
        #
        seq_size_kv = k.shape[-2]
        n_blocks = math.ceil(seq_size_kv / self.block_dim)  
        
        #
        # Pad the keys so that they can be evenly split among blocks
        #
        seq_full_kv = n_blocks * self.block_dim
        pad_dims_kv = (2 * len(k.shape[:-2]) + 1) * (0,) + (seq_full_kv - seq_size_kv,)
        k_pad = F.pad(k, pad_dims_kv)
        v_pad = F.pad(v, pad_dims_kv)

        #
        # Derive the block dimension for x 
        #
        seq_size_x = x.shape[-2]
        block_dim = math.ceil(seq_size_x / n_blocks)
        
        #
        # Pad x so that it can be evenly split
        #
        seq_full_x = n_blocks * block_dim
        pad_dims_x = (2 * len(x.shape[:-2]) + 1) * (0,) + (seq_full_x - seq_size_x,)
        x_pad = F.pad(x, pad_dims_x)

        #
        # Pad the mask and the shift accordingly
        #
        mask_pad = None if mask is None else F.pad(mask, pad_dims_x + pad_dims_kv[-2:])
        shift_pad = None if shift is None else F.pad(shift, pad_dims_x + pad_dims_kv[-2:])

        #
        # Get random permutations of sequence indexes 
        #
        randvals = torch.rand(x_pad.shape[:-1])
        rand_indexes_x = torch.argsort(randvals, dim=-1)

        #
        # Align the permutations with the modeled sequence and its split across heads
        #
        indexes_x_model = rand_indexes_x.unsqueeze(-1).repeat(len(x.shape[:-1]) * (1,) + (self.model_dim,))
        indexes_x_heads = torch.stack(
            torch.chunk(indexes_x_model, self.n_heads, -1)
        )

        #
        # Get random permutations of keys/values
        #
        randvals = torch.rand(k_pad.shape[:-1])
        rand_indexes_kv = torch.argsort(randvals, dim=-1)

        #
        # Align them with the modeled keys and its split across heads
        #
        indexes_kv_model = rand_indexes_kv.unsqueeze(-1).repeat(len(k.shape[:-1]) * (1,) + (self.model_dim,))
        indexes_kv_heads = torch.stack(
            torch.chunk(indexes_kv_model, self.n_heads, -1)
        )

        #
        # Permute inputs to attention
        #
        x_rand = permute_sequence(x_pad, rand_indexes_x)
        k_rand = permute_sequence(k_pad, rand_indexes_kv)
        v_rand = permute_sequence(v_pad, rand_indexes_kv)
        
        mask_rand = None if mask is None else permute_matrix(mask_pad, rand_indexes_x, rand_indexes_kv)
        shift_rand = None if shift is None else permute_matrix(shift_pad, rand_indexes_x, rand_indexes_kv)

        #
        # Split the inputs into blocks and stack them along the first dimension
        #
        x_rand = torch.stack(
            torch.chunk(x_rand, n_blocks, dim=-2)
        )
        k_rand = torch.stack(
            torch.chunk(k_rand, n_blocks, dim=-2)
        )
        v_rand = torch.stack(
            torch.chunk(v_rand, n_blocks, dim=-2)
        )
        mask_rand = None if mask is None else extract_diagonal_blocks(mask_rand, n_blocks, block_dim, self.block_dim)
        shift_rand = None if shift is None else extract_diagonal_blocks(shift_rand, n_blocks, block_dim, self.block_dim)

        #
        # Compute cross-attention with the permuted inputs
        #
        o_rand = super(RandomBlockCrossAttention, self).forward(x_rand, k_rand, v_rand, mask_rand, shift_rand)

        #
        # Reassemble the output
        #
        o_rand = torch.cat(
            torch.chunk(o_rand, n_blocks, 0), dim=-2
        ).squeeze(0)

        #
        # Redistribute the output to the original indexes
        #
        o = torch.scatter(o_rand.clone(), -2, indexes_x_model, o_rand)

        return o[..., :seq_size_x, :] 


def test():
    x = Variable(torch.randn(10,6,32), requires_grad=True)

    sa = SelfAttention(32)
    o, k, v = sa(x)
    print(o.shape, k.shape, v.shape)

    ca = CrossAttention(32)
    o = ca(x, k, v)
    print(o.shape)

    rbsa = RandomBlockSelfAttention(32, 2)
    rbca = RandomBlockCrossAttention(32, 2)


    mask = causal_mask(x)
    shift = alibi_shift(x)


    o, k, v = rbsa(x, mask, shift)


    print(o.shape, k.shape, v.shape)
    x = Variable(torch.randn(10,4,32), requires_grad=True)

    mask = torch.ones(10, 4, 6)
    shift = torch.ones(10, 4, 6)

    o = rbca(x, k, v, mask, shift)

    print(o.shape)


if __name__ == '__main__':
    test()