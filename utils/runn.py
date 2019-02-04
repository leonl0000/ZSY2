"""
Just some stuff for the interpreter
"""
import os
import importlib
import numpy as np
import tensorflow as tf
import pickle

def cls():
    os.system("cls")

def pklLoad(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

reload = importlib.reload

