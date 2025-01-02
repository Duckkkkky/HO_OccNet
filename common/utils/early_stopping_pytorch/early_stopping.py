# early_stopping.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.is_best = False
        
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.is_best = True
        elif loss > self.best_loss + self.delta:
            self.counter += 1
            self.is_best = False
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
            self.is_best = True
