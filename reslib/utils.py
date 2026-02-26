import torch
import torch.nn as nn

def get_orthogonal_matrix(rows: int, cols: int) -> torch.Tensor:
    """
    Generates an orthogonal matrix of size (rows, cols).

    If rows > cols, the columns will be orthogonal.
    If cols > rows, the rows will be orthogonal.
    """
    matrix = torch.empty(rows, cols)
    nn.init.orthogonal_(matrix)
    return matrix
