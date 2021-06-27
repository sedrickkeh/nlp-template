import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def load_data(config):
    data_paths = {'train': config.train_path,
                  'validation': config.valid_path}
    datasets = load_dataset("csv", data_files=data_paths)

    train = datasets["train"]
    valid = datasets["validation"]
    return train, valid


def get_dataloaders(config, tokenizer):
    train, valid = load_data(config)

    # Tokenize
    train = train.map(lambda x: tokenizer(x['text'],
                                            truncation=True, padding='max_length',
                                            max_length = config.max_len), batched=True)
    valid = valid.map(lambda x: tokenizer(x['text'],
                                            truncation=True, padding='max_length',
                                            max_length = config.max_len), batched=True)

    # Set the format for __getitem__ which returns input_ids, token_type_ids , attention_mask and label
    train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'target'])
    valid.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'target'])
    
    trainloader = DataLoader(train, batch_size = config.batch_size)
    validloader = DataLoader(valid, batch_size = config.batch_size)

    return trainloader, validloader