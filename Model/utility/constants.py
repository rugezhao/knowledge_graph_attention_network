import os
from pathlib import Path

GC_HOME = Path(os.environ['HOME'])
SOURCE_PATH = GC_HOME / 'knowledge_graph_attention_network'
SPRINKLER_PATH = GC_HOME / 'sprinkler'

PROJECT_ID = 'cs224w-rec-sys-dff0118132b5'
GC_BUCKET = 'cs224w_sprinkler'
CREDENTIAL_PATH = SPRINKLER_PATH / \
                    'bash_scripts' / 'cs224w-rec-sys-dff0118132b5.json'

EXP_STORAGE = 'experiments'
DATA_STORAGE = 'graph_data'

TEMP_FOLDER = GC_HOME / 'temp_store'
TEMP_FOLDER.mkdir(exist_ok=True)