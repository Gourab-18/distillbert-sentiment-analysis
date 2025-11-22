"""
Day 23: Assignment - Fine-tune BERT for Question Answering
Using Hugging Face Transformers library
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Hugging Face libraries
try:
    from transformers import (
        BertTokenizer, BertForQuestionAnswering,
        DistilBertTokenizer, DistilBertForQuestionAnswering,
        Trainer, TrainingArguments,
        AutoTokenizer, AutoModelForQuestionAnswering,
        pipeline, DefaultDataCollator
    )
    from datasets import Dataset, load_dataset
    import torch
    from torch.utils.data import DataLoader
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Please install transformers: pip install transformers datasets torch")


# ==================== HUGGING FACE BASICS ====================

class HuggingFaceBasics:
    """
    Introduction to Hugging Face library basics
    """

    @staticmethod
    def demonstrate_tokenizer():
        """Demonstrate tokenizer usage"""
        print("=" * 60)
        print("HUGGING FACE TOKENIZER BASICS")
        print("=" * 60)

        if not HF_AVAILABLE:
            print("Hugging Face not available")
            return

        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Sample text
        text = "Hugging Face is creating amazing NLP tools!"

        print(f"\nOriginal text: {text}")

        # Basic tokenization
        tokens = tokenizer.tokenize(text)
        print(f"\n1. Tokenization:")
        print(f"   Tokens: {tokens}")

        # Convert to IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"\n2. Token IDs: {token_ids}")

        # Full encoding (with special tokens)
        encoded = tokenizer.encode(text, add_special_tokens=True)
        print(f"\n3. Full encoding (with [CLS] and [SEP]): {encoded}")

        # Decode back
        decoded = tokenizer.decode(encoded)
        print(f"\n4. Decoded: {decoded}")

        # Encode plus (with attention mask)
        encoding = tokenizer(
            text,
            padding='max_length',
            max_length=20,
            truncation=True,
            return_tensors='pt'
        )
        print(f"\n5. Full encoding output:")
        print(f"   input_ids shape: {encoding['input_ids'].shape}")
        print(f"   attention_mask shape: {encoding['attention_mask'].shape}")
        print(f"   input_ids: {encoding['input_ids'][0].tolist()}")
        print(f"   attention_mask: {encoding['attention_mask'][0].tolist()}")

        return tokenizer

    @staticmethod
    def demonstrate_pipelines():
        """Demonstrate Hugging Face pipelines"""
        print("\n" + "=" * 60)
        print("HUGGING FACE PIPELINES")
        print("=" * 60)

        if not HF_AVAILABLE:
            return

        # 1. Sentiment Analysis
        print("\n1. SENTIMENT ANALYSIS:")
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            result = sentiment_pipeline("I love using Hugging Face transformers!")
            print(f"   Input: 'I love using Hugging Face transformers!'")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"   Error: {e}")

        # 2. Question Answering
        print("\n2. QUESTION ANSWERING:")
        try:
            qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad"
            )
            context = "Hugging Face is a company that develops tools for NLP. They created the transformers library."
            question = "What did Hugging Face create?"
            result = qa_pipeline(question=question, context=context)
            print(f"   Context: '{context}'")
            print(f"   Question: '{question}'")
            print(f"   Answer: '{result['answer']}' (score: {result['score']:.4f})")
        except Exception as e:
            print(f"   Error: {e}")

        # 3. Text Generation
        print("\n3. TEXT GENERATION:")
        try:
            gen_pipeline = pipeline(
                "text-generation",
                model="distilgpt2"
            )
            result = gen_pipeline(
                "Machine learning is",
                max_length=30,
                num_return_sequences=1
            )
            print(f"   Prompt: 'Machine learning is'")
            print(f"   Generated: '{result[0]['generated_text']}'")
        except Exception as e:
            print(f"   Error: {e}")


# ==================== QA DATASET PREPARATION ====================

class QADatasetPreparation:
    """
    Prepare dataset for Question Answering fine-tuning
    """

    @staticmethod
    def create_sample_qa_dataset():
        """Create a sample QA dataset for demonstration"""
        data = {
            "context": [
                "Python is a high-level programming language created by Guido van Rossum. It was first released in 1991. Python is known for its simple syntax and readability.",
                "Machine learning is a subset of artificial intelligence. It allows computers to learn from data without being explicitly programmed. Deep learning is a subset of machine learning.",
                "The transformer architecture was introduced in the paper 'Attention is All You Need' in 2017. It uses self-attention mechanisms and has revolutionized NLP.",
                "Natural Language Processing (NLP) is a field of AI that deals with the interaction between computers and human language. It includes tasks like translation and sentiment analysis.",
                "BERT stands for Bidirectional Encoder Representations from Transformers. It was developed by Google and released in 2018. BERT uses masked language modeling for pre-training."
            ],
            "question": [
                "Who created Python?",
                "What is machine learning a subset of?",
                "When was the transformer architecture introduced?",
                "What does NLP stand for?",
                "What does BERT stand for?"
            ],
            "answer_text": [
                "Guido van Rossum",
                "artificial intelligence",
                "2017",
                "Natural Language Processing",
                "Bidirectional Encoder Representations from Transformers"
            ],
            "answer_start": [
                59,  # Position of "Guido van Rossum" in context
                45,  # Position of "artificial intelligence"
                83,  # Position of "2017"
                0,   # Position of "Natural Language Processing"
                17,  # Position of "Bidirectional..."
            ]
        }
        return pd.DataFrame(data)

    @staticmethod
    def preprocess_qa_data(examples, tokenizer, max_length=384, stride=128):
        """
        Preprocess QA examples for training

        Args:
            examples: Dict with 'context', 'question', 'answer_text', 'answer_start'
            tokenizer: HF tokenizer
            max_length: Maximum sequence length
            stride: Stride for splitting long documents
        """
        questions = examples["question"]
        contexts = examples["context"]

        # Tokenize
        tokenized = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Find start and end positions
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer_start = examples["answer_start"][sample_idx]
            answer_text = examples["answer_text"][sample_idx]
            answer_end = answer_start + len(answer_text)

            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            # Check if answer is in this chunk
            if offsets[context_start][0] > answer_end or offsets[context_end][1] < answer_start:
                tokenized["start_positions"].append(0)
                tokenized["end_positions"].append(0)
            else:
                # Find start token
                idx = context_start
                while idx <= context_end and offsets[idx][0] <= answer_start:
                    idx += 1
                tokenized["start_positions"].append(idx - 1)

                # Find end token
                idx = context_end
                while idx >= context_start and offsets[idx][1] >= answer_end:
                    idx -= 1
                tokenized["end_positions"].append(idx + 1)

        return tokenized


# ==================== BERT QA FINE-TUNING ====================

class BERTQAFineTuner:
    """
    Fine-tune BERT for Question Answering
    """

    def __init__(self, model_name='distilbert-base-uncased', output_dir='./qa_model'):
        """
        Initialize fine-tuner

        Args:
            model_name: Pre-trained model name
            output_dir: Directory to save fine-tuned model
        """
        if not HF_AVAILABLE:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.output_dir = output_dir

        print(f"Loading model: {model_name}")

        # Load tokenizer and model
        if 'distilbert' in model_name.lower():
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"Model loaded on: {self.device}")

    def prepare_dataset(self, train_df, val_df=None):
        """
        Prepare datasets for training

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
        """
        # Convert to HF Dataset
        train_dataset = Dataset.from_pandas(train_df)

        # Tokenize
        def preprocess(examples):
            tokenized = self.tokenizer(
                examples["question"],
                examples["context"],
                max_length=384,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True
            )

            # Simple answer position finding
            start_positions = []
            end_positions = []

            for i in range(len(examples["context"])):
                answer_start = examples["answer_start"][i]
                answer_text = examples["answer_text"][i]
                answer_end = answer_start + len(answer_text)

                # Find token positions
                offsets = tokenized["offset_mapping"][i]

                start_token = 0
                end_token = 0

                for idx, (start, end) in enumerate(offsets):
                    if start <= answer_start < end:
                        start_token = idx
                    if start < answer_end <= end:
                        end_token = idx
                        break

                start_positions.append(start_token)
                end_positions.append(end_token)

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions

            # Remove offset_mapping as it's not needed for training
            tokenized.pop("offset_mapping")

            return tokenized

        self.train_dataset = train_dataset.map(
            preprocess,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        if val_df is not None:
            val_dataset = Dataset.from_pandas(val_df)
            self.val_dataset = val_dataset.map(
                preprocess,
                batched=True,
                remove_columns=val_dataset.column_names
            )
        else:
            self.val_dataset = None

        print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")

    def train(self, num_epochs=3, batch_size=8, learning_rate=3e-5):
        """
        Train the QA model

        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            eval_strategy="epoch" if self.val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if self.val_dataset else False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DefaultDataCollator()
        )

        print("\nStarting training...")
        trainer.train()

        # Save model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"\nModel saved to {self.output_dir}")

    def predict(self, question: str, context: str) -> Dict:
        """
        Make a prediction

        Args:
            question: Question string
            context: Context string

        Returns:
            Dictionary with answer and scores
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)

        # Decode answer
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Get confidence scores
        start_prob = torch.softmax(start_logits, dim=-1)[0, start_idx].item()
        end_prob = torch.softmax(end_logits, dim=-1)[0, end_idx].item()

        return {
            "answer": answer,
            "start_score": start_prob,
            "end_score": end_prob,
            "confidence": (start_prob + end_prob) / 2
        }


# ==================== TEXT SUMMARIZATION ====================

class TextSummarizer:
    """
    Text summarization using transformers
    """

    def __init__(self, model_name='facebook/bart-large-cnn'):
        if not HF_AVAILABLE:
            raise ImportError("Transformers library required")

        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """Generate summary for text"""
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0]['summary_text']


# ==================== MAIN DEMO ====================

def run_qa_demo():
    """Run complete QA demonstration"""

    print("=" * 60)
    print("BERT QUESTION ANSWERING FINE-TUNING")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\nHugging Face libraries not available.")
        print("Install with: pip install transformers datasets torch")
        return

    # 1. Hugging Face Basics
    HuggingFaceBasics.demonstrate_tokenizer()
    HuggingFaceBasics.demonstrate_pipelines()

    # 2. Create sample dataset
    print("\n" + "=" * 60)
    print("CREATING QA DATASET")
    print("=" * 60)

    df = QADatasetPreparation.create_sample_qa_dataset()
    print(f"\nDataset shape: {df.shape}")
    print("\nSample entries:")
    for i in range(min(3, len(df))):
        print(f"\n  Context: {df.iloc[i]['context'][:80]}...")
        print(f"  Question: {df.iloc[i]['question']}")
        print(f"  Answer: {df.iloc[i]['answer_text']}")

    # 3. Initialize fine-tuner
    print("\n" + "=" * 60)
    print("INITIALIZING QA MODEL")
    print("=" * 60)

    try:
        fine_tuner = BERTQAFineTuner(
            model_name='distilbert-base-uncased',
            output_dir='./qa_model_output'
        )

        # 4. Prepare dataset
        train_df = df.iloc[:4]
        val_df = df.iloc[4:]

        fine_tuner.prepare_dataset(train_df, val_df)

        # 5. Train (short demo)
        print("\n" + "=" * 60)
        print("TRAINING MODEL (Demo - 1 epoch)")
        print("=" * 60)

        fine_tuner.train(num_epochs=1, batch_size=2)

        # 6. Test predictions
        print("\n" + "=" * 60)
        print("TESTING PREDICTIONS")
        print("=" * 60)

        test_questions = [
            ("Who created Python?",
             "Python is a high-level programming language created by Guido van Rossum."),
            ("What is deep learning a subset of?",
             "Machine learning is a subset of AI. Deep learning is a subset of machine learning."),
        ]

        for question, context in test_questions:
            result = fine_tuner.predict(question, context)
            print(f"\n  Question: {question}")
            print(f"  Context: {context}")
            print(f"  Answer: {result['answer']}")
            print(f"  Confidence: {result['confidence']:.4f}")

    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        print("Using pre-trained QA pipeline instead...")

        # Fallback to pre-trained pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad"
        )

        print("\nTesting with pre-trained model:")
        context = "Python is a programming language created by Guido van Rossum in 1991."
        question = "Who created Python?"

        result = qa_pipeline(question=question, context=context)
        print(f"  Question: {question}")
        print(f"  Answer: {result['answer']}")
        print(f"  Score: {result['score']:.4f}")

    return fine_tuner if 'fine_tuner' in dir() else None


if __name__ == "__main__":
    model = run_qa_demo()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    This module demonstrated:
    1. Hugging Face tokenizer basics
    2. Using pre-trained pipelines
    3. Creating QA datasets
    4. Fine-tuning BERT for Question Answering
    5. Making predictions with fine-tuned model

    Key concepts:
    - Tokenization with special tokens ([CLS], [SEP])
    - Attention masks for padding
    - Start/end position prediction for extractive QA
    - Transfer learning with pre-trained models
    """)
