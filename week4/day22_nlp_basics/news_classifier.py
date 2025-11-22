"""
Day 22: Assignment - News Article Multi-Class Classifier
Build a complete text classification pipeline for news articles
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLTK for preprocessing
import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class NewsDataGenerator:
    """
    Generate synthetic news dataset for demonstration
    """

    @staticmethod
    def generate_dataset(n_samples_per_class=200):
        """Generate synthetic news articles dataset"""

        categories = {
            'politics': [
                "The government announced new legislation regarding healthcare reform",
                "Senate passed the infrastructure bill with bipartisan support",
                "Presidential campaign focuses on economic policies",
                "Local elections see record voter turnout this year",
                "Congressional hearings examine regulatory oversight",
                "Governor signs executive order on climate initiatives",
                "Political parties debate immigration reform proposals",
                "Supreme Court ruling affects state voting laws",
                "Cabinet members discuss foreign policy strategies",
                "Lawmakers propose tax reform measures"
            ],
            'sports': [
                "Championship game goes into overtime thriller",
                "Star player signs record-breaking contract extension",
                "Team clinches playoff berth with dramatic victory",
                "Coach announces retirement after successful season",
                "Athletes prepare for upcoming Olympic games",
                "Draft picks exceed expectations in debut season",
                "Tournament finals draw millions of viewers",
                "Injury sidelines key player for several weeks",
                "New stadium breaks ground in downtown area",
                "League announces rule changes for next season"
            ],
            'technology': [
                "Tech giant unveils revolutionary smartphone design",
                "Artificial intelligence breakthrough transforms industry",
                "Startup raises millions in venture capital funding",
                "Cybersecurity threats increase for businesses worldwide",
                "New software update improves user experience",
                "Electric vehicle manufacturer expands production",
                "Social media platform announces policy changes",
                "Quantum computing research achieves milestone",
                "Cloud services demand grows exponentially",
                "Robotics company develops autonomous solutions"
            ],
            'business': [
                "Stock market reaches all-time high amid optimism",
                "Company announces quarterly earnings exceed expectations",
                "Merger creates industry-leading conglomerate",
                "Retail sales surge during holiday shopping season",
                "Central bank maintains interest rates steady",
                "Economic indicators suggest continued growth",
                "Unemployment rates drop to historic lows",
                "Trade agreements boost international commerce",
                "Real estate market shows signs of recovery",
                "Corporate profits increase despite challenges"
            ],
            'entertainment': [
                "Blockbuster movie breaks box office records",
                "Music festival announces star-studded lineup",
                "Award-winning actor joins new streaming series",
                "Concert tour sells out within minutes",
                "Film director wins prestigious international award",
                "Television show renewed for multiple seasons",
                "Celebrity couple announces engagement",
                "Album release tops music charts worldwide",
                "Theme park unveils exciting new attractions",
                "Comedy special receives critical acclaim"
            ]
        }

        data = []

        # Templates for generating variations
        templates = [
            "{base}",
            "{base} according to recent reports",
            "Breaking: {base}",
            "Update: {base} sources confirm",
            "Report: {base} analysts say",
            "{base} industry experts predict",
            "News: {base}",
            "Latest: {base}",
            "{base} witnesses report",
            "Developing: {base}"
        ]

        modifiers = [
            "", "today", "this week", "recently", "yesterday",
            "in latest development", "officials say", "report finds"
        ]

        np.random.seed(42)

        for category, base_articles in categories.items():
            count = 0
            while count < n_samples_per_class:
                base = np.random.choice(base_articles)
                template = np.random.choice(templates)
                modifier = np.random.choice(modifiers)

                article = template.format(base=base)
                if modifier and np.random.random() > 0.5:
                    article = f"{article} {modifier}"

                # Add some noise
                if np.random.random() > 0.7:
                    words = article.split()
                    if len(words) > 3:
                        idx = np.random.randint(1, len(words) - 1)
                        words.insert(idx, np.random.choice(['the', 'new', 'major', 'significant', 'important']))
                        article = ' '.join(words)

                data.append({
                    'text': article,
                    'category': category
                })
                count += 1

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        return df


class TextPreprocessor:
    """
    Preprocessing pipeline for text data
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(tokens)

    def preprocess_dataframe(self, df, text_column='text'):
        """Preprocess all texts in a dataframe"""
        df = df.copy()
        df['processed_text'] = df[text_column].apply(self.clean_text)
        return df


class NewsClassifier:
    """
    Multi-class news article classifier
    """

    def __init__(self, vectorizer_type='tfidf', classifier_type='logistic'):
        """
        Initialize classifier

        Args:
            vectorizer_type: 'tfidf' or 'count'
            classifier_type: 'logistic', 'naive_bayes', 'svm', or 'random_forest'
        """
        # Select vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )

        # Select classifier
        classifiers = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'svm': LinearSVC(random_state=42, max_iter=2000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.classifier = classifiers.get(classifier_type, classifiers['logistic'])

        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])

        self.classes = None
        self.preprocessor = TextPreprocessor()

    def fit(self, X_train, y_train, preprocess=True):
        """Train the classifier"""
        if preprocess:
            X_train = [self.preprocessor.clean_text(text) for text in X_train]

        self.pipeline.fit(X_train, y_train)
        self.classes = self.pipeline.classes_
        print(f"Model trained on {len(X_train)} samples")
        print(f"Classes: {self.classes}")

    def predict(self, texts, preprocess=True):
        """Predict categories for texts"""
        if isinstance(texts, str):
            texts = [texts]

        if preprocess:
            texts = [self.preprocessor.clean_text(text) for text in texts]

        return self.pipeline.predict(texts)

    def predict_proba(self, texts, preprocess=True):
        """Get prediction probabilities"""
        if isinstance(texts, str):
            texts = [texts]

        if preprocess:
            texts = [self.preprocessor.clean_text(text) for text in texts]

        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        else:
            # For SVM, use decision function
            decision = self.pipeline.decision_function(texts)
            # Convert to pseudo-probabilities using softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)

    def evaluate(self, X_test, y_test, preprocess=True):
        """Evaluate model performance"""
        if preprocess:
            X_test = [self.preprocessor.clean_text(text) for text in X_test]

        y_pred = self.pipeline.predict(X_test)

        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f}")

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }

    def get_feature_importance(self, top_n=10):
        """Get most important features for each class"""
        feature_names = self.vectorizer.get_feature_names_out()

        if hasattr(self.classifier, 'coef_'):
            coefficients = self.classifier.coef_

            importance_dict = {}
            for i, class_name in enumerate(self.classes):
                if len(coefficients.shape) == 1:
                    # Binary classification
                    class_coef = coefficients
                else:
                    class_coef = coefficients[i]

                top_indices = np.argsort(class_coef)[-top_n:][::-1]
                importance_dict[class_name] = [
                    (feature_names[idx], class_coef[idx])
                    for idx in top_indices
                ]

            return importance_dict
        return None


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to 'confusion_matrix.png'")


def compare_classifiers(X_train, X_test, y_train, y_test):
    """Compare different classifier performance"""
    classifiers = {
        'Logistic Regression + TF-IDF': ('tfidf', 'logistic'),
        'Naive Bayes + Count': ('count', 'naive_bayes'),
        'SVM + TF-IDF': ('tfidf', 'svm'),
        'Random Forest + TF-IDF': ('tfidf', 'random_forest')
    }

    results = []

    for name, (vec_type, clf_type) in classifiers.items():
        print(f"\nTraining: {name}")
        classifier = NewsClassifier(vectorizer_type=vec_type, classifier_type=clf_type)
        classifier.fit(X_train, y_train)

        # Preprocess test data
        X_test_processed = [classifier.preprocessor.clean_text(text) for text in X_test]
        y_pred = classifier.pipeline.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': accuracy
        })
        print(f"  Accuracy: {accuracy:.4f}")

    return pd.DataFrame(results)


def run_news_classifier():
    """Main function to run the news classifier"""

    print("=" * 60)
    print("NEWS ARTICLE MULTI-CLASS CLASSIFIER")
    print("=" * 60)

    # Generate dataset
    print("\n1. GENERATING DATASET")
    print("-" * 40)
    df = NewsDataGenerator.generate_dataset(n_samples_per_class=200)
    print(f"Dataset shape: {df.shape}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())

    # Sample articles
    print("\nSample articles:")
    for category in df['category'].unique()[:3]:
        sample = df[df['category'] == category]['text'].iloc[0]
        print(f"  [{category}]: {sample[:80]}...")

    # Split data
    print("\n2. SPLITTING DATA")
    print("-" * 40)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,
        df['category'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['category'].values
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train and evaluate main classifier
    print("\n3. TRAINING CLASSIFIER")
    print("-" * 40)
    classifier = NewsClassifier(vectorizer_type='tfidf', classifier_type='logistic')
    classifier.fit(X_train, y_train)

    print("\n4. EVALUATING MODEL")
    print("-" * 40)
    results = classifier.evaluate(X_test, y_test)

    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], classifier.classes)

    # Feature importance
    print("\n5. TOP FEATURES PER CATEGORY")
    print("-" * 40)
    importance = classifier.get_feature_importance(top_n=5)
    if importance:
        for category, features in importance.items():
            print(f"\n{category.upper()}:")
            for word, score in features:
                print(f"  {word}: {score:.4f}")

    # Compare classifiers
    print("\n6. COMPARING CLASSIFIERS")
    print("-" * 40)
    comparison = compare_classifiers(X_train, X_test, y_train, y_test)
    print("\nComparison Results:")
    print(comparison.to_string(index=False))

    # Save comparison chart
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(comparison)))
    bars = plt.barh(comparison['Model'], comparison['Accuracy'], color=colors)
    plt.xlabel('Accuracy')
    plt.title('Classifier Comparison')
    plt.xlim(0, 1)
    for bar, acc in zip(bars, comparison['Accuracy']):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{acc:.4f}', va='center')
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=150)
    plt.close()
    print("\nComparison chart saved to 'classifier_comparison.png'")

    # Test predictions
    print("\n7. TEST PREDICTIONS")
    print("-" * 40)
    test_articles = [
        "The president signed a new bill into law today",
        "Team wins championship in thrilling overtime game",
        "New smartphone features advanced AI capabilities",
        "Stock market rallies amid positive earnings reports",
        "Movie premiere attracts major Hollywood celebrities"
    ]

    print("\nPredictions on new articles:")
    for article in test_articles:
        prediction = classifier.predict(article)[0]
        probs = classifier.predict_proba(article)[0]
        confidence = max(probs)
        print(f"\nArticle: '{article[:50]}...'")
        print(f"Prediction: {prediction} (confidence: {confidence:.2%})")

    return classifier, df


if __name__ == "__main__":
    classifier, dataset = run_news_classifier()

    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print("You can now test the classifier with your own articles!")
    print("Example usage:")
    print("  classifier.predict('Your news article here')")
