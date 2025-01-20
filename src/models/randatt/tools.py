import math 
import numpy as np 
import torch 

def sinusoidal_embedding(horizon, embedding_dim):
    """
    Creates a sinusoidal position embedding for sequence of length horizon,
    and of dimensionality embedding_dim.
    """
    #
    # Create an exponential sequence of length embedding_dim / 2
    #
    half_dim = embedding_dim//2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp( (-emb) * torch.arange(half_dim))

    #
    # Create the scaled versions of that sequence for each step under horizon
    #
    steps = torch.arange(horizon)
    emb = steps[:, None] * emb[None, :]

    #
    # Take sine and cosine functions
    #
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    return emb.unsqueeze(0)


def causal_mask(x, until=None):
    """
    Given a batch of sequences of vectors of shape (*, N, D), form a mask 
    that indicates whether the index of the row is no smaller than the column.
    If position until specified, only rows until the position are non-zero.
    """
    N = x.shape[-2]
    
    #
    # Create the mask
    #
    r = torch.arange(N)
    mask = 1. * (r[:, None] >= r[None, :])

    if until is not None:
        mask[until+1:] = 0

    #
    # Align the mask's shape with x
    #
    mask = mask.reshape(len(x.shape[:-2]) * (1,) + 2 * (N,))
    mask = mask.repeat(x.shape[:-2] + 2 * (1,))

    return mask 

def alibi_shift(x):
    """
    Given a batch of sequences of vectors of shape (*, N, D), form a shift
    that will be subtracted from attention logits, following the ALiBi formula.
    """
    N = x.shape[-2]
    
    #
    # Create the shift
    #
    r = torch.arange(N)
    shift = torch.abs(r[:, None] - r[None, :])

    #
    # Align the shift's shape with x
    #
    shift = shift.reshape(len(x.shape[:-2]) * (1,) + 2 * (N,))
    shift = shift.repeat(x.shape[:-2] + 2 * (1,))

    return shift 

def geometric_seq(x, n_seq, coef=math.sqrt(1/2)):
    """
    Given a tensor x, multiply it by a geometric sequence of length
    n_seq and coefficient coef.
    """

    #
    # Repeat x n_seq times
    #
    x_dims = len(x.shape)
    x = x.unsqueeze(0).repeat((n_seq,) + x_dims * (1,))

    #
    # Align the shape of the sequence with x
    #
    seq = coef ** torch.arange(n_seq)
    seq = seq.reshape((n_seq,) + x_dims * (1,))

    #
    # Return the product
    #
    return seq * x


def permute_sequence(x, index):
    """
    Given a batch of sequences of vectors of shape (*, N, D) and indexes of shape (*, N),
    permute the sequences with the indexes, so that output[*, i, :] = x[*, index[*, i], :].
    """
    #
    # Align the dimension of index to that of x
    #
    index = index.unsqueeze(-1)
    index = index.repeat(len(index.shape[:-1]) * (1,) + (x.shape[-1],))

    return torch.gather(x, -2, index)


def permute_matrix(X, left_index, right_index):
    """
    Given a batch of matrices of shape (*, N, M) and indexes of shapes (*, N) and (*, M)
    permute the matrix with the indexes, so that Output[*, i, j] = X[*, left_index[*, i], right_index[*,j]].
    """

    #
    # Align the dimension of index to that of X
    #
    left_index = left_index.unsqueeze(-1)
    left_index = left_index.repeat(len(left_index.shape[:-1]) * (1,) + (X.shape[-2],))

    right_index = right_index.unsqueeze(-2)
    right_index = right_index.repeat(len(right_index.shape[:-2]) * (1,) + (X.shape[-1], 1))

    #
    # Permute the rows of X
    #
    X = torch.gather(X, -2, left_index)

    #
    # Permute the columns of X
    #
    X = torch.gather(X, -1, right_index)

    return X 


def extract_diagonal_blocks(X, n_blocks, block_size1, block_size2=None):
    """
    Given a batch of matrices of shape (*, N * D1, N * D2), extract diagonal blocks of size D.
    End up having N tensors of shape (*, D1, D2).
    """

    block_size2 = block_size1 if block_size2 is None else block_size2
    #
    # Step 1: Reshape the last two dimensions to (N, D1, N, D2)
    #
    X_reshaped = X.view(*X.shape[:-2], n_blocks, block_size1, n_blocks, block_size2)
    #
    # Step 2: Extract the diagonal blocks (along dimension pairs (-4, -2))
    # 
    X_block_diag = X_reshaped.diagonal(dim1=-4, dim2=-2)

    #
    # Step 3: Move the diagonal dimension to the front and reshape if necessary
    # 
    X_block_diag = X_block_diag.permute(-1, *range(X_block_diag.ndim - 1))

    return X_block_diag
