import os
import warnings

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)