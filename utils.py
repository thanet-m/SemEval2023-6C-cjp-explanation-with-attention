import numpy as np


class NpzRepositoryUtils(object):

    @staticmethod
    def save(data, target):
        np.savez(target, data)

    @staticmethod
    def load(source):
        data = np.load(source)
        return data['arr_0']


class VocabRepositoryUtils(object):

    @staticmethod
    def save(data, target):
        np.savetxt(target, data, fmt='%s')

    @staticmethod
    def load(source):
        return np.loadtxt(source, dtype=str, comments=None)


