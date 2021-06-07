import json
import pickle

from datetime import datetime
from pathlib import Path


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return json.JSONEncoder.default(obj)


def json_load(file):
    with open(file, "r") as f:
        return json.load(f)


def json_dump(obj, file, **kwargs):
    with open(file, "w") as f:
        return json.dump(obj, f, default=_json_default, **kwargs)


def pickle_load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def pickle_dump(obj, file):
    with open(file, "wb") as f:
        return pickle.dump(obj, f)
