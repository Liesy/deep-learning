import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def main():
    '''程序入口'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--arg1', type=int, default=1)
    parser.add_argument('--arg2', type=float, default=2.0)

    args = parser.parse_args()


if __name__ == '__main__':
    main()