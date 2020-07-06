from torch.nn import L1Loss, MSELoss
from .losses import MahalanobisLoss
from .nets import SimpleAutoencoder

__all__ = {'MahalanobisLoss', 'SimpleAutoencoder', 'L1Loss', 'MSELoss'}
