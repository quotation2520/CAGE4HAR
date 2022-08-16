from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import numpy as np
import os


def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def record_result(result, epoch, best_acc, best_f1, c_mat):
    """ Record evaluation results."""
    if epoch >= 0:
        result.write('Best validation epoch | accuracy: {:.2f}%, F1: {:.4f} (at epoch {})\n'.format(best_acc, best_f1, epoch))
    elif epoch == -1:
        result.write('\n\nTest (Best) | accuracy: {:.2f}%, F1: {:.4f}\n'.format(best_acc, best_f1))
    elif epoch == -2:
        result.write('\n\nTest (Final) | accuracy: {:.2f}%, F1: {:.4f}\n'.format(best_acc, best_f1))
    elif epoch == -3:
        result.write('\n\nFinal validation epoch | accuracy: {:.2f}%, F1: {:.4f}\n'.format(best_acc, best_f1))
    
    result.write(np.array2string(c_mat))
    result.flush()
    result.close
