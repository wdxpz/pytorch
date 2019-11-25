import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
Model_DIR = os.path.join(BASE_DIR, 'model')

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(Model_DIR):
    os.mkdir(Model_DIR)


TextClassifier_Config = {
    'workers': 4,
    'ngpu': 2,
    'ngrams': 2,
    'embed_size': 16,
    'batch_size': 16,
    'lr': 4,
    'train_valid_ratio': 0.8,
    'num_epochs': 5,
}