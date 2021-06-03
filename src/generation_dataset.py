import json
from pathlib import Path

from torch.utils.data import Dataset

from utils.logger import logger


class GenerationDataset(Dataset):
    def __init__(self, json_file: Path):
        self.data = [json.loads(line) for line in json_file.read_text().split("\n")[:-1]]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def overview(self):
        logger.info(f"Size: {len(self)}")
        logger.info(f"Keys: {sorted(self[0].keys())}")
        logger.info(f"The first entry: {json.dumps(self[0], indent=2, ensure_ascii=False)}")
