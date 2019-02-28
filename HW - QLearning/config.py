import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import gym
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import tqdm

ENVIRONMENT = gym.make('Breakout-v0').unwrapped
ENVIRONMENT.reset()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print("Is python : {}".format(is_ipython))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : {}".format(DEVICE))


AMOUNT_OF_ACTIONS = ENVIRONMENT.action_space.n
print("Number of actions : {}".format(AMOUNT_OF_ACTIONS))






REPLAY_MEMORY = 50000
BATCH_SIZE = 64
INPUT_SIZE = 84
AMOUNT_OF_EPISODES = 50000
HELPER_UPDATE = 100
DISCOUNT_FACTOR = 0.9
