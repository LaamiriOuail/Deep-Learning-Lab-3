# Lab 3

## Overview
## Dependencies

## Part 1 Classification Task:
### 1. By using scrapping libraries (Scrapy / BeautifulSoup), try to collect text data from several Arabic web site concerning one topic then prepare your Dataset as Below:


| Text                                                                  | Score         |
|-----------------------------------------------------------------------|---------------|
| Accuracy                                                              | 0.9911        |
| F1 Score                                                              | 0.9887        |
| Loss                                                                  | 0.0145        |
| Training Time                                                         | 15.08s        |









## Part 2 Transformer (Text generation):
###  Step 1: Import Necessary Libraries and Initialize Model
```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import os
import csv
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import logging
import warnings

# Set device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)

# Utility function to choose from top probabilities
def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)
```

### Step 2: Define Custom Dataset
```python
from google.colab import drive

drive.mount('/content/drive')

# Define the path where you want to create the folder


class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path='jokes_data/'):
        super().__init__()

        short_jokes_path = os.path.join(jokes_dataset_path, 'shortjokes.csv')

        self.joke_list = []
        self.end_of_text_token = ""

        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]

dataset = JokesDataset("./drive/My Drive/Colab Notebooks/data")
joke_loader = DataLoader(dataset, batch_size=1, shuffle=True)
```

### Step 3: Define Hyperparameters
```python
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
```

### Step 4: Model Training
```python
# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None
# Train loop
for epoch in range(EPOCHS):
    for idx, joke in enumerate(joke_loader):
        joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)

        if joke_tens.size()[1] > MAX_SEQ_LEN:
            continue

        if not torch.is_tensor(tmp_jokes_tens):
            tmp_jokes_tens = joke_tens
            continue
        else:
            if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                work_jokes_tens = tmp_jokes_tens
                tmp_jokes_tens = joke_tens
            else:
                tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                continue

        outputs = model(work_jokes_tens, labels=work_jokes_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch
    torch.save(model.state_dict(), f"./drive/My Drive/Colab Notebooks/models/gpt2_medium_joker_{epoch}.pt")
```

### Step 5: Generating Jokes
```python
MODEL_EPOCH = EPOCHS-1
model_path = f"./drive/My Drive/Colab Notebooks/models/gpt2_medium_joker_{MODEL_EPOCH}.pt"
model.load_state_dict(torch.load(model_path))

jokes_output_file_path = f'./drive/My Drive/Colab Notebooks/data/generated_{MODEL_EPOCH}.txt'

model.eval()
if os.path.exists(jokes_output_file_path):
    os.remove(jokes_output_file_path)

joke_num = 0
with torch.no_grad():
    for joke_idx in range(1000):
        joke_finished = False
        cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)

            if next_token_id in tokenizer.encode(''):
                joke_finished = True
                break

        if joke_finished:
            joke_num = joke_num + 1
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

            with open(jokes_output_file_path, 'a') as f:
                f.write(f"{output_text} \n\n")
                print(f"{output_text} \n\n")
```
