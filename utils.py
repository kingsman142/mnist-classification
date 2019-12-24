import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", action = 'store', default = 100, type = int)
parser.add_argument("--batch-size", action = 'store', default = 64, type = int)
parser.add_argument("--learning-rate", action = 'store', default = 0.0002, type = float)
parser.add_argument("--optim", action = 'store', default = "sgd", type = str)
args = parser.parse_args()
