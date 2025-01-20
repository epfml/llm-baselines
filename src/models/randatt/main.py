import torch
from model import Encoder, Decoder

def main():
    # Configuration for the transformer
    config = {
        'version': 'Light',        # Use 'Light' version of blocks
        'n_blocks': 2,             # Number of encoder/decoder layers
        'alibi': False,            # Use sinusoidal embeddings (not ALiBi)
        'drop_emb': 0.1,           # Embedding dropout
        'model_dim': 32,           # Embedding dimension
        'block_dim': 8,            # Block size for random attention
        'n_heads': 2               # Number of attention heads
    }

    # Create encoder and decoder
    encoder = Encoder(**config)
    decoder = Decoder(**config)

    # Input tensor: Shape (batch_size, sequence_length, model_dim)
    batch_size = 4
    sequence_length = 10
    input_tensor = torch.randn(batch_size, sequence_length, config['model_dim'])

    # Forward pass through encoder
    encoded_output, keys, values = encoder(input_tensor)
    print(f"Encoded Output Shape: {encoded_output.shape}")

    # Forward pass through decoder
    decoded_output = decoder(input_tensor, keys, values)
    print(f"Decoded Output Shape: {decoded_output.shape}")

if __name__ == "__main__":
    main()
