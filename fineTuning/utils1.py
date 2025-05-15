import os
import csv
import json
import numpy as np
import torch
import pytorch_lightning as pl
from pyserini.search import LuceneSearcher  # Uncomment if you use this

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mydata, seq_length, tokenizer):
        self.labels = np.array(mydata['label'])
        self.seq1 = np.array(mydata['input'])
        self.seq2 = np.array(mydata['output'])
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        seq1 = self.seq1[idx]
        seq2 = self.seq2[idx]
        
        item1 = self.tokenizer(seq1, truncation=True, max_length=self.seq_length, padding='max_length')
        item2 = self.tokenizer(seq2, truncation=True, max_length=self.seq_length, padding='max_length')
        
        input_ids = torch.tensor(item1.input_ids)
        attention_mask = torch.tensor(item1.attention_mask)
        labels = torch.tensor(item2.input_ids)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }

    def __len__(self):
        return len(self.labels)


class MyModel(pl.LightningModule):
    def __init__(self, model, device, tokenizer):
        super(MyModel, self).__init__()
        self.model = model
        self.mydevice = device
        self.tokenizer = tokenizer
        self.mystep = 0
        self._train_losses = []

    def training_step(self, batch, batch_nb):
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        self._train_losses.append(loss.detach().cpu().item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self._train_losses:
            avg_loss = np.mean(self._train_losses)
            self.log('train_epoch_loss', avg_loss, prog_bar=True)
            self._train_losses = []
        self.save_model()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.mystep += 1
        if self.mystep % 500 == 0:
            directory = f'monoT5-bin-plus/chk/model-{self.mystep}'
            os.makedirs(directory, exist_ok=True)
            os.makedirs(f'{directory}/out2', exist_ok=True)

            print(f'saving model -------- {self.mystep}')
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), f'{directory}/pytorch_model.bin')
            model_to_save.config.to_json_file(f'{directory}/config.json')
            self.tokenizer.save_pretrained(directory)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        input_ids = batch['input_ids'].to(self.mydevice)
        attention_mask = batch['attention_mask'].to(self.mydevice)
        labels = batch['label'].to(self.mydevice)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        self.model.train()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.25)
        return [optimizer], [{'scheduler': scheduler, 'name': 'log_lr'}]

    def save_model(self):
        directory = 'monoT5-bin-plus/chk/model'
        os.makedirs(directory, exist_ok=True)
        os.makedirs(f'{directory}/out2', exist_ok=True)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), f'{directory}/pytorch_model.bin')
        model_to_save.config.to_json_file(f'{directory}/config.json')
        self.tokenizer.save_pretrained(directory)


class MyUtils():
    def __init__(self):
        self.searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')  # Uncomment if needed
        pass

    def get_query(self, id):
        file_topic = 'baseline/topics.dl20.tsv'
        with open(file_topic) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for row in read_tsv:
                query_id, query_text = row[0], row[1]
                if query_id == id:
                    return query_text
        return 'query not found'

    def get_doc(self, id):
        doc = self.searcher.doc(id)
        return json.loads(doc.raw())['contents']

    def gen_prompt(self, query, doc):
        return f'Is the question: "{query}" answered by the document: "{doc}"? Give an explanation.'
