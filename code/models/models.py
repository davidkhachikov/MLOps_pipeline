import torch
import torch.nn as nn

class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, vocab):
        super(TextClassificationModel, self).__init__()
        self.vocab = vocab
        self.embedding = nn.EmbeddingBag(len(vocab), 256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )


    def forward(self, batch, device):
        text_list, offsets = [], [0]
        for text in batch:
            text_list.append(text)
            offsets.append(len(text))

        text_list = sum(text_list, [])
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
        numbers = torch.tensor(self.vocab(text_list), dtype=torch.int64).to(device)
        embedded = self.embedding(input=numbers, offsets=offsets)
        logits = self.classifier(embedded)
        return logits
