import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from knn import *


def main():
    '''程序入口'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, default=1)

    args = parser.parse_args()

    knn_classifier = KNN(args.k)
    


if __name__ == '__main__':
    
    main()