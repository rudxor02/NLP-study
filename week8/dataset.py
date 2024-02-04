import json

from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, json_path: str, keys_to_delete: list[str] = []):
        json_dict = json.load(open(json_path, "r"))
        if keys_to_delete:
            for key in keys_to_delete:
                del json_dict[key]
        self.items = list(json_dict.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]
