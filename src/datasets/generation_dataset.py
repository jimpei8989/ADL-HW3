import json
from pathlib import Path

from torch.utils.data import Dataset

from utils.logger import logger


class GenerationDataset(Dataset):
    @classmethod
    def from_jsonl(cls, jsonl_file: Path, **kwargs):
        return cls(
            [json.loads(line) for line in jsonl_file.read_text().split("\n")[:-1]], **kwargs
        )

    def __init__(self, data, tokenizer=None, content_max_length=384, title_max_length=96):
        self.data = data
        self.tokenizer = tokenizer

        self.content_max_length = content_max_length
        self.title_max_length = title_max_length

        self.size = None

    def __getitem__(self, index):
        content = self.data[index]["maintext"]
        title = self.data[index]["title"]

        if self.tokenizer is None:
            return {
                "input_str": content,
                "target_str": title,
            }
        else:
            content = self.tokenizer(
                content,
                max_length=self.content_max_length,
                padding="max_length",
                truncation=True,
            )
            with self.tokenizer.as_target_tokenizer():
                title = self.tokenizer(
                    title,
                    max_length=self.title_max_length,
                    padding="max_length",
                    truncation=True,
                )

            return {**content, "labels": title.input_ids}

    def __len__(self):
        if self.size is None:
            return len(self.data)
        else:
            return min(self.size, len(self.data))

    def overview(self):
        logger.info(f"Size: {len(self)}")
        logger.info(f"Keys: {sorted(self[0].keys())}")
        logger.info(f"The first entry: {json.dumps(self[0], indent=2, ensure_ascii=False)}")
