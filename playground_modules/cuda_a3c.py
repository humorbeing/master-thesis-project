from util import *
# from model_v1 import Model
from model_v2 import Model
# from trainer import train_v1 as train
from trainer import train_v2 as train
num_processes = 5
lr = 0.0001
if __name__ == "__main__":
    run(Model, train, test, num_processes, lr)