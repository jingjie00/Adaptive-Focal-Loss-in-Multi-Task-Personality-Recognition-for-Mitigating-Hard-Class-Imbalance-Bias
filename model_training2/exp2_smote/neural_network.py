import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from prutils.prdataset import MBTIDataset, EssaysDataset
from prutils.prevaluation import PrEvaluation, print_performance
from sklearn.feature_extraction.text import TfidfVectorizer
from prutils.prtracking import TrackingManager


# Define a function to reset random seeds
def reset_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

reset_random_seed()

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

class PersonalityClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_classes):
        super(PersonalityClassifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First layer
            nn.ReLU(),  # Activation
            nn.Linear(hidden_dim, number_of_classes)  # Output layer
        )

    def forward(self, features):
        logits = self.mlp(features)
        return logits


def train_step(tracker, train_loader, personality_classifier, loss_fn, optimizer):
    personality_classifier.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = personality_classifier(features)
        loss = loss_fn(logits, labels)
        tracker.training_push(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(personality_classifier.parameters(), 1.0)
        optimizer.step()

def validation_step(tracker, validation_loader, personality_classifier, loss_fn):
    personality_classifier.eval()
    for features, labels in validation_loader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = personality_classifier(features)
            loss = loss_fn(logits, labels)
            tracker.validation_push(loss.item())

def evaluation_step(evaluation_loader, personality_classifier):
    personality_classifier.eval()
    evaluator = PrEvaluation()
    for features, labels in evaluation_loader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = personality_classifier(features)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()

            # rearrange accorind to dimension
            predictions = predictions.T
            labels = labels.T

            predictions = predictions.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            evaluator.push(predictions, labels)

    return evaluator.get_performance_metrics()


def run(dataset_name):
    print(f"Running {dataset_name}...")

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    if dataset_name == "mbti":
        trainset = MBTIDataset(device=device, settype="train", tfidf_vectorizer=tfidf_vectorizer)
        validationset = MBTIDataset(device=device, settype="validation",tfidf_vectorizer=tfidf_vectorizer)
        evaluationset = MBTIDataset(device=device, settype="test",tfidf_vectorizer=tfidf_vectorizer)
        input_dim = 5000  # Assuming TF-IDF features
        number_of_classes = 4
    else:
        trainset = EssaysDataset(device=device, settype="train",tfidf_vectorizer=tfidf_vectorizer)
        validationset = EssaysDataset(device=device, settype="validation",tfidf_vectorizer=tfidf_vectorizer)
        evaluationset = EssaysDataset(device=device, settype="test",tfidf_vectorizer=tfidf_vectorizer)
        input_dim = 5000  # Assuming TF-IDF features
        number_of_classes = 5

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validationset, batch_size=32, shuffle=False)
    evaluation_loader = DataLoader(evaluationset, batch_size=32, shuffle=False)

    print(f"Data loaded: {len(trainset)} training samples, {len(validationset)} validation samples, {len(evaluationset)} evaluation samples")

    personality_classifier = PersonalityClassifier(input_dim=input_dim, hidden_dim=128, number_of_classes=number_of_classes)
    personality_classifier.to(device)

    # Change to Focal Loss
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(personality_classifier.parameters(), lr=5e-4)

    tracker = TrackingManager(f"{dataset_name}/")

    epochs = 100
    for epoch in range(epochs):
        train_step(tracker, train_loader, personality_classifier, loss_fn, optimizer)
        validation_step(tracker, validation_loader, personality_classifier, loss_fn)
        performance = evaluation_step(evaluation_loader, personality_classifier)

        info = tracker.go_to_new_epoch(personality_classifier)
        tl = info["training_loss"]
        vl = info["validation_loss"]
        mark = info["mark"]

        print(f"Epoch {epoch+1}/{epochs} {mark} - TL:{tl:.4f}, VL:{vl:.4f}")
        print_performance(performance)

if __name__ == "__main__":
    run("essays")
    run("mbti")
