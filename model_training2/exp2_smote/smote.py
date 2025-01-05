from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import numpy as np
from prutils.prevaluation import PrEvaluation, print_performance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings('ignore')

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=3,C=0.5,n_jobs=-1),
    "SVM": LinearSVC(C=0.15),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=50),
    "Decision Tree": DecisionTreeClassifier(max_depth=15)
}

class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word)>2]
    
def run(dataset_name):
    print(f"Running {dataset_name}...")
    # Load dataset
    if dataset_name == "essays":
        dataset = load_dataset("jingjietan/essays-big5")
        dimension = ["O", "C", "E", "A", "N"]
    else:
        dataset = load_dataset("jingjietan/kaggle-mbti")  # Replace with your dataset
        dimension = ["O", "C", "E", "A"]
    
    for dim in dimension:
        print(f"======Running for dimension: {dim}")
        train_data = dataset['train']
        test_data = dataset['test']

        # Select the columns
        X_train = train_data['text']  # Text column in training set
        y_train = np.array(train_data[dim])  # Target column (replace 'O' with the actual column name if different)

        X_test = test_data['text']  # Text column in test set
        y_test = np.array(test_data[dim])  # Target column

        # Convert text to TF-IDF features
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english',tokenizer=Lemmatizer())  # Using unigrams and bigrams
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # # Apply SMOTE to oversample the minority class
        smote = SMOTE(random_state=0)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)


        y_test = y_test.tolist()
        y_test = [int(i) for i in y_test]

        for clf_name, clf in classifiers.items():
            print(f"-----Performance for {clf_name}:")
            clf.fit(X_train_resampled, y_train_resampled)

            # Make predictions and evaluate the model
            y_pred = clf.predict(X_test_tfidf)

            # Convert numpy array to list for PrEvaluation compatibility
            y_pred = y_pred.tolist()

            # Convert all elements in list to int
            y_pred = [int(i) for i in y_pred]

            evaluator = PrEvaluation()
            evaluator.push([y_pred], [y_test])
            performance = evaluator.get_performance_metrics()
            print_performance(performance)


# if main
if __name__ == "__main__":
    run("mbti")
    run("essays")