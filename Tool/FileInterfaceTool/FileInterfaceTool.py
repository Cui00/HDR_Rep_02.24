#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh & lh'
__date__ = '14/01/2016'
__version__ = '2.0'

import yaml
import pickle as pk
import time
import os
# import numpy as np
import glob


def loadYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return yaml.load(stream=file)

def loadAllYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return yaml.load_all(stream=file)

def dumpYaml(fileName, data, method='w'):
    with open(fileName, method) as file:
        yaml.dump(data=data, stream=file)

def dumpAllYaml(data, fileName, method='w'):
    with open(fileName, method) as file:
        yaml.dump_all(documents=data, stream=file)

def pkLoad(fileName, method='r'):
    with open(fileName, method) as File:
        return pk.load(File)

def isExist(fileName):
    return os.path.exists(fileName)

def createFile(fileName):
    if not isExist(fileName):
        os.system('mkdir ' + fileName)

def globPath(path):
    return glob.glob(path)

def joinPath(path, paths):
    return os.path.join(path, paths)

def absPath(path):
    return os.path.abspath(path)

def isFile(path):
    if os.path.isfile(path):
        return True
    return False

def isDir(path):
    if os.path.isdir(path):
        return True
    return False


if __name__ == '__main__':
    Path = r'./'
    print globPath(Path)