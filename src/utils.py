"""
This module contains utility functions for the project. It includes functions for data loading, preprocessing, visualization, model evaluation, training...
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import numpy as np
from collections import Counter

import typing

import warnings

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms.v2 as transforms

from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")


def set_seed(seed: int, device: str='cpu'):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(cm: np.ndarray, class_names: list):
    """
    Plot the confusion matrix.

    Args:
        cm (array-like): The confusion matrix to plot.
        class_names (list): The list of class names corresponding to the confusion matrix.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_history(history: dict[str, list[float]]):
    """
    Plot the training history.

    Args:
        history (dict): A dictionary containing the training history, with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def load_data(data_dir: str, batch_size: int, transform: transforms.Compose) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the dataset and create data loaders.

    Args:
        data_dir (str): The directory where the dataset is located.
        batch_size (int): The batch size for the data loaders.
        transform (transforms.Compose): The transformations to apply to the images.

    Returns:
        tuple: A tuple containing the training and validation data loaders.
    """
    train_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str, criterion) -> typing.Tuple[float, float, list[int], list[int]]:
    """
    Evaluate the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader to evaluate on.
        device (str): The device to use for evaluation ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the loss, accuracy, true labels, and predicted labels.
    """
    model.eval()
    criterion = criterion
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_true_labels, all_pred_labels

def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, device: str, criterion, optimizer, num_epochs: int) -> dict[str, list[float]]:
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (str): The device to use for training ('cpu' or 'cuda').
        criterion: The loss function to use for training.
        optimizer: The optimizer to use for training.
        num_epochs (int): The number of epochs to train for.

    Returns:
        dict: A dictionary containing the training history, with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, device, criterion)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    return history

def save_model(model: nn.Module, path: str):
    """
    Save the model to the specified path.

    Args:
        model (nn.Module): The model to save.
        path (str): The path where the model should be saved.
    """
    torch.save(model.state_dict(), path)
    
def load_model(model: nn.Module, path: str, device: str):
    """
    Load the model from the specified path.

    Args:
        model (nn.Module): The model architecture to load the weights into.
        path (str): The path where the model weights are saved.
        device (str): The device to load the model onto ('cpu' or 'cuda').
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    
def get_class_names(data_dir: str) -> list[str]:
    """
    Get the class names from the dataset directory.

    Args:
        data_dir (str): The directory where the dataset is located.

    Returns:
        list: A list of class names.
    """
    train_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train')
    return train_dataset.classes

def get_device() -> str:
    """
    Get the available device for computation.

    Returns:
        str: The device to use for computation ('cuda' if available, otherwise 'cpu').
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def show_results(true_labels: list[int], pred_labels: list[int], class_names: list[str]):
    """
    Show the evaluation results including confusion matrix and classification report.

    Args:
        true_labels (list): The true labels of the samples.
        pred_labels (list): The predicted labels of the samples.
        class_names (list): The list of class names corresponding to the labels.
    """

    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, class_names)

    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

