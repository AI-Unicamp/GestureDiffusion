import numpy as np
import json

class Logger:
    def __init__(self, name):
        self.name = name
        self.loss = []
        self.min_loss = np.inf
        self.acc = []
        
    def logbatch(self, loss, acc):
        self.loss.append(loss)
        self.acc.append(acc)
        if loss < self.min_loss:
            self.min_loss = loss
            self.acc_at_min_loss = acc

    def save(self, path):
        # Save json file with loss and acc and min_loss at epoch
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)