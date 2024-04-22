import torch

class SegmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, segment_length):
        self.dataset = dataset
        self.segment_length = segment_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"] if "attention_mask" in item else None
        labels = item["labels"] if "labels" in item else None

        segments = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in range(0, len(input_ids), self.segment_length):
            segments["input_ids"].append(input_ids[i : i + self.segment_length])
            if attention_mask is not None:
                segments["attention_mask"].append(attention_mask[i : i + self.segment_length])
            if labels is not None:
                segments["labels"].append(labels[i : i + self.segment_length])

        return segments