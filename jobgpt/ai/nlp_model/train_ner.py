"""
NER Model Training Script for JobGPT

This script trains a custom spaCy Named Entity Recognition (NER) model
to identify key entities in CVs/resumes like skills, education, years of experience, etc.
"""

import os
import random
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NERTrainer:
    """Train a custom spaCy NER model for CV/resume parsing."""
    
    def __init__(self, model_name: str = "en_core_web_md", output_dir: str = "./output"):
        """
        Initialize the NER trainer.
        
        Args:
            model_name: Base spaCy model to start with
            output_dir: Directory to save the trained model
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load base model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded base model: {model_name}")
        except OSError:
            logger.info(f"Downloading {model_name}...")
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
    
    def prepare_training_data(self, training_data: List[Dict[str, Any]]) -> List[Example]:
        """
        Prepare training data for spaCy.
        
        Args:
            training_data: List of dictionaries with text and annotations
            
        Returns:
            List of spaCy Example objects
        """
        examples = []
        
        # Create training examples
        for item in training_data:
            text = item["text"]
            annotations = item["annotations"]
            doc = self.nlp.make_doc(text)
            
            # Convert annotations to spaCy format
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
                else:
                    logger.warning(f"Skipping entity due to span error: {text[start:end]} ({start}, {end})")
            
            # Set entities in the document
            example = Example.from_dict(doc, {"entities": [(e.start_char, e.end_char, e.label_) for e in ents]})
            examples.append(example)
        
        return examples
    
    def train(self, training_data: List[Dict[str, Any]], n_iter: int = 30, dropout: float = 0.2) -> None:
        """
        Train the NER model.
        
        Args:
            training_data: List of dictionaries with text and annotations
            n_iter: Number of training iterations
            dropout: Dropout rate for training
        """
        # Prepare the pipe
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        
        # Add entity labels
        entity_types = set()
        for item in training_data:
            for _, _, label in item["annotations"]:
                entity_types.add(label)
                
        for entity_type in entity_types:
            ner.add_label(entity_type)
        
        # Prepare training examples
        examples = self.prepare_training_data(training_data)
        
        # Only train NER
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        # Train the model
        with self.nlp.disable_pipes(*other_pipes):
            # Reset weights
            if examples:
                self.nlp.begin_training()
                
                # Training loop
                for i in range(n_iter):
                    random.shuffle(examples)
                    losses = {}
                    
                    # Batch training with increasing batch size
                    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                    for batch in batches:
                        self.nlp.update(batch, drop=dropout, losses=losses)
                    
                    # Log progress
                    logger.info(f"Iteration {i+1}/{n_iter}, Losses: {losses}")
        
        # Save the model
        self.nlp.to_disk(self.output_dir / "cv_ner_model")
        logger.info(f"Model saved to {self.output_dir / 'cv_ner_model'}")
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            test_data: List of dictionaries with text and annotations
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load the trained model
        trained_model_path = self.output_dir / "cv_ner_model"
        if not trained_model_path.exists():
            logger.error("No trained model found for evaluation")
            return {"precision": 0, "recall": 0, "f1": 0}
        
        try:
            nlp_trained = spacy.load(trained_model_path)
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            return {"precision": 0, "recall": 0, "f1": 0}
        
        # Evaluation metrics
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        # Evaluate on test data
        for item in test_data:
            text = item["text"]
            gold_entities = set([(start, end, label) for start, end, label in item["annotations"]])
            
            # Get predictions
            doc = nlp_trained(text)
            pred_entities = set([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
            
            # Calculate metrics
            tp += len(gold_entities & pred_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    @staticmethod
    def create_training_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create sample training and test data for CV/resume NER.
        
        Returns:
            Tuple of training data and test data
        """
        # Sample training data for CV parsing
        training_data = [
            {
                "text": "John Smith has 5 years of experience in Python and JavaScript.",
                "annotations": [
                    (23, 30, "EXPERIENCE_YEARS"),
                    (46, 52, "SKILL"),
                    (57, 67, "SKILL")
                ]
            },
            {
                "text": "BSc in Computer Science from Stanford University, graduated in 2015.",
                "annotations": [
                    (0, 24, "EDUCATION"),
                    (30, 49, "ORGANIZATION"),
                    (65, 69, "DATE")
                ]
            },
            {
                "text": "Proficient in React, Node.js, and AWS. Certified AWS Solutions Architect.",
                "annotations": [
                    (14, 19, "SKILL"),
                    (21, 28, "SKILL"),
                    (34, 37, "SKILL"),
                    (39, 68, "CERTIFICATION")
                ]
            },
            {
                "text": "Worked as a Senior Software Engineer at Google for 3 years.",
                "annotations": [
                    (11, 35, "JOB_TITLE"),
                    (39, 45, "ORGANIZATION"),
                    (50, 57, "EXPERIENCE_YEARS")
                ]
            },
            {
                "text": "Skills include Python, SQL, Tableau, and machine learning.",
                "annotations": [
                    (15, 21, "SKILL"),
                    (23, 26, "SKILL"),
                    (28, 35, "SKILL"),
                    (41, 57, "SKILL")
                ]
            },
            {
                "text": "Masters in Data Science from MIT with focus on deep learning.",
                "annotations": [
                    (0, 24, "EDUCATION"),
                    (30, 33, "ORGANIZATION"),
                    (48, 61, "SKILL")
                ]
            },
            {
                "text": "Completed the Google Cloud Architect certification in January 2021.",
                "annotations": [
                    (14, 43, "CERTIFICATION"),
                    (59, 72, "DATE")
                ]
            },
            {
                "text": "Experienced in agile methodologies with 7+ years in software development.",
                "annotations": [
                    (16, 35, "SKILL"),
                    (41, 49, "EXPERIENCE_YEARS"),
                    (53, 73, "SKILL")
                ]
            }
        ]
        
        # Test data (similar structure but different examples)
        test_data = [
            {
                "text": "Jane Doe has 10 years of experience in Java and Docker.",
                "annotations": [
                    (13, 21, "EXPERIENCE_YEARS"),
                    (37, 41, "SKILL"),
                    (46, 52, "SKILL")
                ]
            },
            {
                "text": "MSc in Artificial Intelligence from UC Berkeley, completed in 2018.",
                "annotations": [
                    (0, 28, "EDUCATION"),
                    (34, 44, "ORGANIZATION"),
                    (60, 64, "DATE")
                ]
            }
        ]
        
        return training_data, test_data


if __name__ == "__main__":
    # Create trainer
    trainer = NERTrainer(output_dir="./model_output")
    
    # Generate training data
    training_data, test_data = NERTrainer.create_training_data()
    
    # Train the model
    trainer.train(training_data, n_iter=10)  # Reduced iterations for demonstration
    
    # Evaluate the model
    metrics = trainer.evaluate(test_data)
    print(f"Evaluation results - Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}") 