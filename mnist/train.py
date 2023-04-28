
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from pathlib import Path

# PyTorch model
class MyModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(base_directory, train_path, validation_path, epochs=50, batch_size=32, learning_rate=0.01):
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train.drop(X_train.columns[-1], axis=1, inplace=True)

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation.drop(X_validation.columns[-1], axis=1, inplace=True)

    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = TensorDataset(torch.tensor(X_validation.values, dtype=torch.float32), torch.tensor(y_validation.values, dtype=torch.long))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = MyModel(input_size=X_train.shape[1], hidden_size=128, output_size=10)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0.0

        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)
            loss.backward()

            optimizer.step()

            train_loss += loss.item() * x_batch.shape[0]
            train_acc += accuracy_score(y_batch.numpy(), np.argmax(y_pred.detach().numpy(), axis=1)) * x_batch.shape[0]

        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        validation_loss = 0.0
        validation_acc = 0.0

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                y_pred = model(x_batch)

                loss = loss_fn(y_pred, y_batch)

                validation_loss += loss.item() * x_batch.shape[0]
                validation_acc += accuracy_score(y_batch.numpy(), np.argmax(y_pred.numpy(), axis=1)) * x_batch.shape[0]

            validation_loss /= len(validation_dataset)
            validation_acc /= len(validation_dataset)

        print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, Validation loss: {validation_loss:.4f}, Validation accuracy: {validation_acc:.4f}")

    model_filepath = Path(base_directory) / "model" / "001"
    torch.save(model.state_dict(), model_filepath)

if __name__ == "__main__":
    # Any hyperparameters provided by the training job are passed to the entry point
    # as script arguments. SageMaker will also provide a list of special parameters
    # that you can capture here. Here is the full list:
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_directory", type=str, default="/opt/ml/")
    parser.add_argument("--train_path", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", None))
    parser.add_argument("--validation_path", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", None))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args, _ = parser.parse_known_args()

    train(
        base_directory=args.base_directory,
        train_path=args.train_path,
        validation_path=args.validation_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
