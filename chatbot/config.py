# Default word tokens
import os

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

#corpus file
CORPUS_NAME = "cornell movie-dialogs corpus"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
CORPUS_DIR = os.path.join(DATA_DIR, CORPUS_NAME)

#movie data fields
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

#trim the vocabulary
MIN_COUNT = 3
MAX_LENGTH = 10


ChatBotConfig = {
    'model_name': 'cb_model',
    'load_saved_model': True,
    'attn_method': 'dot', #'dot', 'general', 'concat'
    'hidden_size': 500,
    'encoder_n_layers': 2,
    'decoder_n_layers': 2,
    'dropout': 0.1,
    'batch_size': 64,
    'gradient_clip': 50.0,
    'teacher_forcing_ratio': 1.0,
    'learning_rate': 0.0001,
    'decoder_learning_ratio': 5.0,
    'checkpoint_iter': 40000,
    'n_iteration': 4000,
    'print_every': 100,
    'save_every': 1000
}