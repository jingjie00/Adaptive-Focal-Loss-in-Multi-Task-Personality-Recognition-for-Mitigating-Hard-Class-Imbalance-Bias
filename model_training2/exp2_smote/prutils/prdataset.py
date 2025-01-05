import re
import nltk
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from datasets import load_dataset

nltk.download('punkt')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation and digits
    text = re.sub(r"[-/@.?!_,:;()|0-9]", "", text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

class EssaysDataset(Dataset):
    def __init__(self, device, settype, tfidf_vectorizer=None, max_features=5000):
        self.device = device
        self.settype = settype
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer(max_features=max_features)
        self.essays_corpus = None
        self.process()

    def process(self):
        # Load and preprocess the dataset
        essays_corpus = load_dataset("jingjietan/essays-big5", split=self.settype).map(
            lambda x: {
                'processed_text': preprocess_text(x['text']),
                'labels': [float(x["O"]), float(x["C"]), float(x["E"]), float(x["A"]), float(x["N"])]
            }
        )

        # Fit or transform TF-IDF vectorizer
        tfidf_matrix = (
            self.tfidf_vectorizer.fit_transform(essays_corpus['processed_text'])
            if self.settype == 'train'
            else self.tfidf_vectorizer.transform(essays_corpus['processed_text'])
        )

        # Store the processed tensor and labels (no need for the text)
        self.essays_corpus = {
            'feature': tfidf_matrix.toarray(),  # Convert to dense array
            'labels': torch.tensor(essays_corpus['labels'], dtype=torch.float32).to(self.device)
        }

    def __len__(self):
        return len(self.essays_corpus['labels'])

    def __getitem__(self, index):
        return (
            torch.tensor(self.essays_corpus['feature'][index], dtype=torch.float32).to(self.device),
            self.essays_corpus['labels'][index]
        )

class MBTIDataset(Dataset):
    def __init__(self, device, settype, tfidf_vectorizer=None, max_features=5000):
        self.device = device
        self.settype = settype
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer(max_features=max_features)
        self.mbti_corpus = None
        self.process()

    def process(self):
        # Load and preprocess the dataset
        mbti_corpus = load_dataset("jingjietan/kaggle-mbti", split=self.settype).map(
            lambda x: {
                'processed_text': preprocess_text(x['text']),
                'labels': [float(x["O"]), float(x["C"]), float(x["E"]), float(x["A"])]
            }
        )

        # Fit or transform TF-IDF vectorizer
        tfidf_matrix = (
            self.tfidf_vectorizer.fit_transform(mbti_corpus['processed_text'])
            if self.settype == 'train'
            else self.tfidf_vectorizer.transform(mbti_corpus['processed_text'])
        )

        # Store the processed tensor and labels (no need for the text)
        self.mbti_corpus = {
            'feature': tfidf_matrix.toarray(),  # Convert to dense array
            'labels': torch.tensor(mbti_corpus['labels'], dtype=torch.float32).to(self.device)
        }

    def __len__(self):
        return len(self.mbti_corpus['labels'])

    def __getitem__(self, index):
        return (
            torch.tensor(self.mbti_corpus['feature'][index], dtype=torch.float32).to(self.device),
            self.mbti_corpus['labels'][index]
        )





if __name__ == "__main__":
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)

    # Create the training dataset
    trainset = MBTIDataset(device=device, settype='train', tfidf_vectorizer=tfidf_vectorizer)

    # Access a sample from the dataset
    sample = trainset[0]


    # trainset = MBTIDataset(device=device, settype="train", tfidf_vectorizer=tfidf_vectorizer)
    # sample = trainset[0]

    print("TF-IDF Features:", sample[0])  # TF-IDF feature vector
    print("Labels:", sample[1])  # Multi-label tensor

