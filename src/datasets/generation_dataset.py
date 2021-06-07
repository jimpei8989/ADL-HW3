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

    def __init__(
        self,
        data,
        tokenizer=None,
        tokenizer_return_tensors=None,
        content_max_length=384,
        title_max_length=96,
        include_id=False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer_return_tensors = tokenizer_return_tensors

        self.content_max_length = content_max_length
        self.title_max_length = title_max_length
        self.include_id = include_id

        self.size = None
        self.has_label = "title" in self.data[0]

    def __getitem__(self, index):
        ret = {}
        if self.include_id:
            ret |= {"id": self.data[index]["id"]}

        content = self.data[index]["maintext"]
        title = self.data[index].get("title", None)

        if self.tokenizer is None:
            ret |= {
                "input_str": content,
                "target_str": title,
            }
        else:
            ret |= self.tokenizer(
                content,
                max_length=self.content_max_length,
                padding="max_length",
                truncation=True,
                return_tensors=self.tokenizer_return_tensors
            )
            if title is not None:
                with self.tokenizer.as_target_tokenizer():
                    ret |= {
                        "labels": self.tokenizer(
                            title,
                            max_length=self.title_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors=self.tokenizer_return_tensors,
                        ).input_ids
                    }
        return ret

    def __len__(self):
        if self.size is None:
            return len(self.data)
        else:
            return min(self.size, len(self.data))

    def overview(self):
        logger.info(f"Size: {len(self)}")
        logger.info(f"Keys: {sorted(self[0].keys())}")
        logger.info(f"The first entry: {json.dumps(self[0], indent=2, ensure_ascii=False)}")
