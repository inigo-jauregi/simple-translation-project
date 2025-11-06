from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl


class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, src_lang="eng_Latn", tgt_lang="zho_Hans", max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['translation']['en']
        tgt_text = item['translation']['zh']

        # Tokenize source
        self.tokenizer.src_lang = self.src_lang
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target
        self.tokenizer.src_lang = self.tgt_lang
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = tgt_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=8, max_samples=10000):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_samples = max_samples

    def setup(self, stage=None):
        # Load WMT19 zh-en dataset
        dataset = load_dataset("wmt19", "zh-en", split="train")

        # Take subset for faster training
        dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        # Split into train/val
        split = dataset.train_test_split(test_size=0.1, seed=42)

        self.train_dataset = TranslationDataset(split['train'], self.tokenizer)
        self.val_dataset = TranslationDataset(split['test'], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )