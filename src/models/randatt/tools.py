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
    
    device = x.device  # Ensure the mask is created on the same device

    #
    # Create the mask
    #
    r = torch.arange(N, device=device)
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
    device = x.device
    #
    # Create the shift
    #
    r = torch.arange(N, device=device)
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
    index = index.to(x.device)

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
    device = X.device
    # assert X.size(-2) == left_index.size(-2), "Mismatch in rows"
    # assert X.size(-1) == right_index.size(-1), "Mismatch in columns"
    left_index = left_index.to(device)
    right_index = right_index.to(device)

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



def geometric_cumsum(v, gamma):
    """
    Compute the exponentially decaying weighted cumulative sum.

    Args:
        v: Tensor of shape (batch_size, seq_length, embedding_dim)
        gamma: Decay factor, scalar or tensor broadcastable to (batch_size, seq_length)

    Returns:
        Tensor of shape (batch_size, seq_length, embedding_dim)
    """
    seq_len = v.shape[1]

    #
    # Create a geometric sequence (gamma, gamma^2, ..., gamma^seq_len)
    #
    weights = torch.cumprod(torch.ones((seq_len,), device=v.device) * gamma, dim=0) + 1e-9

    #
    # Reverse the sequence so that it is (gamma^seq_len, ..., gamma^2, gamma^1)
    #
    weights = weights.flip(0).unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_length, 1)
    weights = weights.clamp(1e-9, 1)

    #
    # Compute the (unnormalized) weighted cumulative sum of v: (gamma^seq_len v1, gamma^seq_len v1 + gamma^{seq_len-1} v2, ..., gamma^seq_len v1 + ... + gamma vseq_len)
    #
    cumsum_v = torch.cumsum(v * weights, dim=1)

    #
    # Compute the normalizing constants (gamma^seq_len, gamma^seq_len + gamma^{seq_len-1}, ..., gamma^seq_len + ... + gamma)
    #
    normalizers = torch.cumsum(weights, dim=1)

    return cumsum_v / normalizers

def softmax_cumsum(v, L):
    """
    Compute the softmax-weighted causal cumulative sum.

    Args:
        v: Tensor of shape (batch_size, seq_length, embedding_dim)
        L: Tensor of shape (batch_size, seq_length, embedding_dim), unnormalized weights.

    Returns:
        Tensor of shape (batch_size, seq_length, embedding_dim)
    """
    #
    # Exponentiate L to obtain positive weights
    #
    L_exp = torch.exp(L).clamp(min=1e-9, max=1e6)

    #
    # Compute the (unnormalized) weighted cumulative sum of v: (l1*v1, l1*v1 + l2*v2, ...)
    #
    cumsum_v = torch.cumsum(L_exp * v, dim=1)

    #
    # Compute the cumulative sum of the weights: (l1, l1 + l2, l1 + l2 + l3, ...)
    #
    cumsum_weights = torch.cumsum(L_exp, dim=1).clamp(min=1e-9)

    #
    # Normalize the cumulative sum by dividing by the cumulative weights
    #
    return cumsum_v / cumsum_weights

