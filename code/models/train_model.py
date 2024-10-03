import os
import ast
import nltk
import yaml
import torch
import mlflow
import importlib
import pandas as pd
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from mlflow.tracking import MlflowClient
from models import TextClassificationModel
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_train_test(config):
    path_to_train = os.path.join(config['BASE_DIR'], config['TRAIN_TEST_PATH'], 'train.csv')
    path_to_test = os.path.join(config['BASE_DIR'], config['TRAIN_TEST_PATH'], 'test.csv')

    train = pd.read_csv(path_to_train)
    test = pd.read_csv(path_to_test)

    train['text'] = train['text'].apply(lambda x: ast.literal_eval(x))
    test['text'] = test['text'].apply(lambda x: ast.literal_eval(x))
    return train, test


def get_vocab(df: pd.DataFrame):
    def yield_tokens(df):
        for _, sample in df.iterrows():
            yield sample.to_list()[0]

    vocab = build_vocab_from_iterator(yield_tokens(df), specials=special_symbols)
    vocab.set_default_index(UNK_IDX)
    return vocab


def get_train_test_dataloaders(train: pd.DataFrame, test: pd.DataFrame):
    def collate_batch(batch):
        label_list, text_list = [], []
        for _text, _label in batch:
            label_list.append(_label)
            text_list.append(_text)
        label_list = [int(float_num) for float_num in label_list]
        label_list = torch.tensor(label_list, dtype=torch.int64)
        
        return label_list.to(device), text_list

    train_dataloader = DataLoader(
        train.to_numpy(), batch_size=128, shuffle=True, collate_fn=collate_batch
    )

    test_dataloader = DataLoader(
        test.to_numpy(), batch_size=128, shuffle=False, collate_fn=collate_batch
    )

    return train_dataloader, test_dataloader


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    for i, batch in loop:
        labels, texts = batch
        model.zero_grad()
        outputs = model(texts, device)
        # print(outputs.shape)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        loop.set_postfix({"loss": train_loss/(i * len(labels))})
    if scheduler is not None:
        scheduler.step()

def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    ckpt_path='best.pth'
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for _, batch in loop:
            labels, texts = batch
            outputs = model(texts, device)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs, dim=1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()
            loop.set_postfix({"loss": val_loss / total, "acc": correct / total})

        accuracy = correct / total
        loss_final = val_loss / total
        print(loss_final, best_so_far)
        if loss_final < best_so_far:
            best_so_far = loss_final
            torch.save(model, ckpt_path)

    return loss_final, accuracy


def train_model(
    model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    scheduler, 
    loss_fn, 
    epochs,
    model_path
):
    best = float('inf')
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch_num=epoch)
        best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best, ckpt_path=model_path)


def setup_mlflow_experiment(experiment_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()

    # Check if the experiment already exists
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # If no experiment exists, create a new one
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    elif experiment.lifecycle_stage == 'deleted':
        # If the experiment is deleted, restore it
        print(f"Experiment '{experiment_name}' is deleted. Restoring it...")
        client.restore_experiment(experiment.experiment_id)
        experiment = client.get_experiment(experiment.experiment_id)
    else:
        # Experiment is already active
        print(f"Experiment '{experiment_name}' is active and ready to use.")

    return experiment


def run_experiment(name, cfg, model, train_dataloader, val_dataloader, loss_fn, epochs, model_path, best=float('inf')):
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = setup_mlflow_experiment(name)

    params = cfg['optimizer']['params']
    module_name = cfg['optimizer']['module_name']
    class_name  = cfg['optimizer']['class_name']
    class_instance = getattr(importlib.import_module(module_name), class_name)
    optimizer = class_instance(model.parameters(), **params)

    params = cfg['scheduler']['params']
    module_name = cfg['scheduler']['module_name']
    class_name  = cfg['scheduler']['class_name']
    class_instance = getattr(importlib.import_module(module_name), class_name)
    scheduler = class_instance(optimizer, **params)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=name):
        mlflow.set_tag("hyperparameters", str(cfg))

        for key, value in cfg.items():
            mlflow.log_param(key, value)

        for epoch in range(epochs):
            mlflow.log_metric("epoch", epoch)
            train_one_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch_num=epoch)
            loss, acc = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best, ckpt_path=model_path)

            mlflow.log_metric("validation_loss", loss)
            mlflow.log_metric("validation_accuracy", acc)

            if loss < best:
                best = loss
        best_model = torch.load(model_path)
        mlflow.pytorch.log_model(best_model, f'{experiment}_model')
        print(f"Experiment '{name}' completed. Best validation loss: {best}")
    mlflow.end_run()
    return best

def train():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

    with open("./configs/main.yml", 'r') as file:
        config = yaml.safe_load(file)

    train, test = get_train_test(config)
    vocab = get_vocab(train)
    train_dataloader, val_dataloader = get_train_test_dataloaders(train, test)

    with open("./configs/mlflow.yml", 'r') as file:
        experiment_config = yaml.safe_load(file)

    best = float('inf')
    for params in experiment_config:
        epochs = config['train']['epochs']
        model = TextClassificationModel(3, vocab).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        model_path = config['TRAINED_MODEL_PATH']
        name = params['name']
        best = run_experiment(
            name,
            params,
            model,
            train_dataloader,
            val_dataloader,
            loss_fn,
            epochs,
            model_path,
            best
        )


if __name__ == '__main__':
    train()