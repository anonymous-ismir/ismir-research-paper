import math
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pydub import AudioSegment
import torchaudio
from IPython.display import Audio


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

