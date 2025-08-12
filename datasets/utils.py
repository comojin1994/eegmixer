from huggingface_hub import hf_hub_download
from utils.pbs_utils import HUGGINGFACE_REPO_DICT
from utils.setup_utils import get_huggingface_token
import numpy as np
import sqlite3


def connect_to_database(db_path):
    """Connect to SQLite database."""
    return sqlite3.connect(db_path)


def fetch_metadata(cursor, query):
    """Fetch metadata based on a query."""
    cursor.execute(query)
    return cursor.fetchall()


def load_pbs_scores(model, dataset):
    """Load PBS scores from Hugging Face hub."""
    token = get_huggingface_token()
    repo_info = HUGGINGFACE_REPO_DICT[model][dataset]["repo"]
    pbs_score_file = hf_hub_download(
        repo_id=repo_info, filename="pbs/pbs.npy", token=token
    )
    return np.load(pbs_score_file)
