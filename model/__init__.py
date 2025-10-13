import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import torch.optim as optim
from torchmetrics import Accuracy

from transformers import AutoTokenizer

import math
import random
import os
from modules.model_utils import *
from modules.replay_buffer import CausalReplayBuffer
from pydantic import BaseModel