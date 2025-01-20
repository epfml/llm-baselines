import torch 
import torch.nn as nn
from torch.autograd import Variable

import src.models.randatt.tools

from blocks import EncoderBlock, LightEncoderBlock, DecoderBlock, LightDecoderBlock

class Encoder(nn.Module):

    def __init__(self, **kwargs):

        super().__init__()

        #
        # Read off the block version ('' or 'Light')
        #
        version = kwargs.pop('version')
        self_name = self.__class__.__name__
        block_cls = globals()[version + self_name + 'Block']

        #
        # Read off the esential hyper-parameters
        #
        self.n_blocks = kwargs.pop('n_blocks')
        self.alibi = kwargs.pop('alibi')
        drop_emb = kwargs.pop('drop_emb')
        self.drop_emb = nn.Dropout(drop_emb)

        #
        # Build the transformer blocks
        #
        self.blocks = nn.Sequential()
        for _ in range(self.n_blocks):
            self.blocks.append(
                block_cls(**kwargs)
            )

        self.model_dim = kwargs.pop('model_dim')

    def forward(self, x):
        #
        # Read off the shape of the input sequence
        #
        horizon, model_dim = x.shape[-2:]
        
        #
        # Embed the position sinusoidally or with ALiBi
        #
        pos_emb = None if self.alibi else tools.sinusoidal_embedding(horizon, model_dim)
        pos_shift = tools.alibi_shift(x) if self.alibi else None
        x = self.drop_emb(x + pos_emb) if pos_emb is not None else x

        #
        # Initialize the keys and values to be returned by the encoder
        #
        keys = torch.empty((self.n_blocks,) + x.shape)
        values = torch.empty((self.n_blocks,) + x.shape)

        #
        # Apply the blocks
        #
        for b, block in enumerate(self.blocks):
            x, keys[b], values[b] = block(x, mask=None, shift=pos_shift)

        return x, keys, values
    

class Decoder(Encoder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, keys, values, until=None):

        #
        # Create the causal mask
        #
        mask = tools.causal_mask(x, until)

        #
        # Read off the shape of the input sequence
        #
        horizon, model_dim = x.shape[-2:]
        
        #
        # Embed the position sinusoidally or with ALiBi
        #
        pos_emb = None if self.alibi else tools.sinusoidal_embedding(horizon, model_dim)
        pos_shift = tools.alibi_shift(x) if self.alibi else None
        x = self.drop_emb(x + pos_emb) if pos_emb is not None else x

        #
        # Apply the blocks
        #
        for b, block in enumerate(self.blocks):
            x = block(x, keys[b], values[b], mask, pos_shift)

        return x
        

def test():
    
    kwargs = {
        'version' : 'Light',
        'n_blocks' : 2,
        'alibi' : False, # cross attention doesn't support alibi
        'drop_emb': 0.1,
        'model_dim': 32,
        'block_dim': 8,
        'n_heads' : 2
    }

    enc = Encoder(**kwargs)
    dec = Decoder(**kwargs)

    x = Variable(torch.randn(5, 12, 32), requires_grad=True)

    o, k, v = enc(x)

    print(o.shape, k.shape, v.shape)

    o = dec(x, k, v)
    
    print(o.shape)

if __name__ == '__main__':

    test()