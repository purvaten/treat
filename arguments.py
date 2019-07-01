"""Arguments for training model."""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-data", "--data", default="cliparts,letters", type=str, help="comma separated data types, choose among cliparts, letters")
parser.add_argument("-cliparts_dir", "--cliparts_dir", type=str, help="Path to directory containing clipart images")
parser.add_argument("-letters_dir", "--letters_dir", type=str, help="Path to directory containing letter images")
parser.add_argument("-alpha", "--alpha", default=0.25, type=float, help="Value of interpolation constant")
parser.add_argument("-zsize", "--zsize", default=128, type=int, help="Value of intermediate representation size")
parser.add_argument("-datalimit", "--datalimit", default=40250, type=int, help="Limit of data to be loaded")
parser.add_argument("-batchsize", "--batchsize", default=100, type=int, help="Batch size")
parser.add_argument("-id", "--id", default='job_0', type=str, help="Job ID")
parser.add_argument(
    "-model", "--model", default="alexnet", type=str, help="Model of the encoder",
    choices=["alexnet", "bigresnet", "smallresnet"]
)
