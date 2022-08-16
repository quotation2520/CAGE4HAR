import os
import sys
import numpy as np
import csv
import random
import pandas as pd
import scipy.io
from scipy.signal import butter, lfilter
from collections import Counter
from . import *


def butterworth_filter(data, fs, lowcut=20, order=3):
    b, a = butter(order, lowcut, fs=fs)
    y = lfilter(b, a, data)
    return y
