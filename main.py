import argparse

from knn import *


def main():
    '''程序入口'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int)

    args = parser.parse_args()

    knn_classifier = KNN()



if __name__ == '__main__':
    
    main()