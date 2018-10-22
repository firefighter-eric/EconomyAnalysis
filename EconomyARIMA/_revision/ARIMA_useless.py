import numpy as np
import tensorflow as tf


class ARIMA:
    def __init__(self, p, d, q, data):
        self.p = p
        self.s = d
        self.q = q
        self.data = data
        self.ar = self.auto_regressive()

    def auto_regressive(self):
        ar = list()
        ar.append(self.data)
        for i in range(self.p):
            ar.append(np.diff(ar[-1]))
        return ar

    def integrated(self):
        pass

