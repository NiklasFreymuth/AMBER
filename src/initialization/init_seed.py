import random

import numpy as np
import torch
from pytorch_lightning import seed_everything


def initialize_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    seed_everything(seed)
