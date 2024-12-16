import torch

# Define hyperparameters and constants
FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/taiwan_clean_dataset.csv"
CHECKPOINTS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/checkpoints"
LOGS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/logs"
TARGET_COLUMN = "AverageTemperature"
BATCH_SIZE = 32
N_SPLITS = 5
LR = 1e-3
NUM_EPOCHS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQUENCE_NO = 36