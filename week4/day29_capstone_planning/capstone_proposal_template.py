"""
Day 29: Capstone Project Planning
Template and tools for defining an end-to-end ML project
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import os


# ==================== PROJECT PROPOSAL TEMPLATE ====================

class CapstoneProjectProposal:
    """
    Template for capstone project proposal
    """

    def __init__(self):
        self.proposal = {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "project_info": {},
            "problem_definition": {},
            "dataset": {},
            "methodology": {},
            "architecture": {},
            "evaluation": {},
            "deployment": {},
            "timeline": {},
            "risks": [],
            "deliverables": []
        }

    def set_project_info(self, title: str, author: str, description: str,
                         domain: str, project_type: str):
        """
        Set basic project information

        Args:
            title: Project title
            author: Your name
            description: Brief project description
            domain: Project domain (e.g., NLP, Computer Vision, Time Series)
            project_type: Type (Classification, Regression, Generation, etc.)
        """
        self.proposal["project_info"] = {
            "title": title,
            "author": author,
            "description": description,
            "domain": domain,
            "project_type": project_type
        }

    def define_problem(self, problem_statement: str, business_value: str,
                       target_users: List[str], success_criteria: List[str]):
        """
        Define the problem to be solved

        Args:
            problem_statement: Clear statement of the problem
            business_value: Why this problem matters
            target_users: Who will benefit from the solution
            success_criteria: Measurable success criteria
        """
        self.proposal["problem_definition"] = {
            "problem_statement": problem_statement,
            "business_value": business_value,
            "target_users": target_users,
            "success_criteria": success_criteria
        }

    def define_dataset(self, name: str, source: str, size: str,
                       features: List[str], target: str,
                       preprocessing_steps: List[str],
                       potential_biases: List[str] = None):
        """
        Define the dataset

        Args:
            name: Dataset name
            source: Where to obtain the data
            size: Number of samples
            features: List of features
            target: Target variable
            preprocessing_steps: Required preprocessing
            potential_biases: Known or potential biases
        """
        self.proposal["dataset"] = {
            "name": name,
            "source": source,
            "size": size,
            "features": features,
            "target": target,
            "preprocessing_steps": preprocessing_steps,
            "potential_biases": potential_biases or []
        }

    def define_methodology(self, approach: str, algorithms: List[str],
                           baseline_model: str, advanced_model: str,
                           techniques: List[str]):
        """
        Define the methodology

        Args:
            approach: Overall approach (supervised, unsupervised, etc.)
            algorithms: Algorithms to try
            baseline_model: Simple baseline for comparison
            advanced_model: Main model to develop
            techniques: Key techniques (transfer learning, augmentation, etc.)
        """
        self.proposal["methodology"] = {
            "approach": approach,
            "algorithms": algorithms,
            "baseline_model": baseline_model,
            "advanced_model": advanced_model,
            "techniques": techniques
        }

    def define_architecture(self, components: Dict[str, str],
                            pipeline_steps: List[str],
                            infrastructure: str):
        """
        Define system architecture

        Args:
            components: Key components and their purpose
            pipeline_steps: ML pipeline steps
            infrastructure: Required infrastructure
        """
        self.proposal["architecture"] = {
            "components": components,
            "pipeline_steps": pipeline_steps,
            "infrastructure": infrastructure
        }

    def define_evaluation(self, primary_metric: str, secondary_metrics: List[str],
                          evaluation_strategy: str, target_performance: Dict[str, float]):
        """
        Define evaluation strategy

        Args:
            primary_metric: Main metric to optimize
            secondary_metrics: Additional metrics
            evaluation_strategy: How to evaluate (cross-validation, etc.)
            target_performance: Target values for metrics
        """
        self.proposal["evaluation"] = {
            "primary_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
            "evaluation_strategy": evaluation_strategy,
            "target_performance": target_performance
        }

    def define_deployment(self, deployment_platform: str, api_type: str,
                          monitoring_strategy: str, scaling_requirements: str):
        """
        Define deployment plan

        Args:
            deployment_platform: Where to deploy (cloud, local, etc.)
            api_type: REST, gRPC, etc.
            monitoring_strategy: How to monitor in production
            scaling_requirements: Expected load and scaling needs
        """
        self.proposal["deployment"] = {
            "deployment_platform": deployment_platform,
            "api_type": api_type,
            "monitoring_strategy": monitoring_strategy,
            "scaling_requirements": scaling_requirements
        }

    def add_risk(self, risk: str, impact: str, mitigation: str):
        """Add a risk to the proposal"""
        self.proposal["risks"].append({
            "risk": risk,
            "impact": impact,
            "mitigation": mitigation
        })

    def add_deliverable(self, deliverable: str, description: str):
        """Add a deliverable"""
        self.proposal["deliverables"].append({
            "deliverable": deliverable,
            "description": description
        })

    def save(self, filepath: str):
        """Save proposal to JSON"""
        self.proposal["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(filepath, 'w') as f:
            json.dump(self.proposal, f, indent=2)
        print(f"Proposal saved to {filepath}")

    def load(self, filepath: str):
        """Load proposal from JSON"""
        with open(filepath, 'r') as f:
            self.proposal = json.load(f)
        print(f"Proposal loaded from {filepath}")

    def display(self):
        """Display the proposal"""
        print("\n" + "=" * 70)
        print("CAPSTONE PROJECT PROPOSAL")
        print("=" * 70)

        # Project Info
        info = self.proposal.get("project_info", {})
        print(f"\n{'='*70}")
        print(f"PROJECT: {info.get('title', 'Untitled')}")
        print(f"{'='*70}")
        print(f"Author: {info.get('author', 'Unknown')}")
        print(f"Domain: {info.get('domain', 'N/A')}")
        print(f"Type: {info.get('project_type', 'N/A')}")
        print(f"\nDescription:\n{info.get('description', 'No description')}")

        # Problem Definition
        problem = self.proposal.get("problem_definition", {})
        print(f"\n{'-'*70}")
        print("PROBLEM DEFINITION")
        print(f"{'-'*70}")
        print(f"\nProblem Statement:\n{problem.get('problem_statement', 'N/A')}")
        print(f"\nBusiness Value:\n{problem.get('business_value', 'N/A')}")
        print(f"\nTarget Users: {', '.join(problem.get('target_users', []))}")
        print(f"\nSuccess Criteria:")
        for criterion in problem.get('success_criteria', []):
            print(f"  • {criterion}")

        # Dataset
        dataset = self.proposal.get("dataset", {})
        print(f"\n{'-'*70}")
        print("DATASET")
        print(f"{'-'*70}")
        print(f"Name: {dataset.get('name', 'N/A')}")
        print(f"Source: {dataset.get('source', 'N/A')}")
        print(f"Size: {dataset.get('size', 'N/A')}")
        print(f"Target: {dataset.get('target', 'N/A')}")
        print(f"\nFeatures: {', '.join(dataset.get('features', [])[:5])}...")
        print(f"\nPreprocessing Steps:")
        for step in dataset.get('preprocessing_steps', []):
            print(f"  {step}")

        # Methodology
        method = self.proposal.get("methodology", {})
        print(f"\n{'-'*70}")
        print("METHODOLOGY")
        print(f"{'-'*70}")
        print(f"Approach: {method.get('approach', 'N/A')}")
        print(f"Baseline: {method.get('baseline_model', 'N/A')}")
        print(f"Advanced Model: {method.get('advanced_model', 'N/A')}")
        print(f"\nAlgorithms: {', '.join(method.get('algorithms', []))}")
        print(f"Techniques: {', '.join(method.get('techniques', []))}")

        # Evaluation
        evaluation = self.proposal.get("evaluation", {})
        print(f"\n{'-'*70}")
        print("EVALUATION")
        print(f"{'-'*70}")
        print(f"Primary Metric: {evaluation.get('primary_metric', 'N/A')}")
        print(f"Secondary: {', '.join(evaluation.get('secondary_metrics', []))}")
        print(f"Strategy: {evaluation.get('evaluation_strategy', 'N/A')}")
        print(f"\nTarget Performance:")
        for metric, target in evaluation.get('target_performance', {}).items():
            print(f"  {metric}: {target}")

        # Deployment
        deploy = self.proposal.get("deployment", {})
        print(f"\n{'-'*70}")
        print("DEPLOYMENT")
        print(f"{'-'*70}")
        print(f"Platform: {deploy.get('deployment_platform', 'N/A')}")
        print(f"API Type: {deploy.get('api_type', 'N/A')}")
        print(f"Monitoring: {deploy.get('monitoring_strategy', 'N/A')}")

        # Risks
        risks = self.proposal.get("risks", [])
        if risks:
            print(f"\n{'-'*70}")
            print("RISKS")
            print(f"{'-'*70}")
            for i, risk in enumerate(risks, 1):
                print(f"\n{i}. {risk.get('risk', 'N/A')}")
                print(f"   Impact: {risk.get('impact', 'N/A')}")
                print(f"   Mitigation: {risk.get('mitigation', 'N/A')}")

        # Deliverables
        deliverables = self.proposal.get("deliverables", [])
        if deliverables:
            print(f"\n{'-'*70}")
            print("DELIVERABLES")
            print(f"{'-'*70}")
            for d in deliverables:
                print(f"  • {d.get('deliverable', 'N/A')}: {d.get('description', '')}")

        print("\n" + "=" * 70)


# ==================== SAMPLE CAPSTONE PROJECTS ====================

def create_sentiment_analysis_proposal():
    """
    Sample: Sentiment Analysis Capstone Project
    """
    proposal = CapstoneProjectProposal()

    proposal.set_project_info(
        title="Real-Time Social Media Sentiment Analysis System",
        author="ML Student",
        description="""
        Build an end-to-end sentiment analysis system that processes social media
        posts in real-time, classifies sentiment (positive/negative/neutral),
        and provides insights through an interactive dashboard.
        """,
        domain="Natural Language Processing",
        project_type="Multi-class Classification"
    )

    proposal.define_problem(
        problem_statement="""
        Businesses need to understand customer sentiment on social media to
        respond quickly to issues and track brand perception. Manual monitoring
        is impossible at scale.
        """,
        business_value="""
        - Real-time brand monitoring
        - Early detection of PR crises
        - Customer feedback analysis
        - Competitive intelligence
        """,
        target_users=["Marketing teams", "Customer service", "Product managers"],
        success_criteria=[
            "Accuracy > 85% on sentiment classification",
            "Process > 100 posts per second",
            "Response latency < 100ms",
            "Dashboard updates in real-time"
        ]
    )

    proposal.define_dataset(
        name="Twitter Sentiment Dataset + Custom Data",
        source="Kaggle Sentiment140 + Twitter API",
        size="1.6M tweets + 50K custom labeled",
        features=["text", "timestamp", "user_id", "hashtags", "mentions"],
        target="sentiment (positive/negative/neutral)",
        preprocessing_steps=[
            "1. Text cleaning (URLs, mentions, special chars)",
            "2. Tokenization and lemmatization",
            "3. Handle emojis and slang",
            "4. Sequence padding/truncation"
        ],
        potential_biases=["Twitter demographic bias", "English-only", "Topic bias"]
    )

    proposal.define_methodology(
        approach="Supervised Learning with Transfer Learning",
        algorithms=["DistilBERT", "RoBERTa", "LSTM baseline"],
        baseline_model="TF-IDF + Logistic Regression",
        advanced_model="Fine-tuned DistilBERT",
        techniques=[
            "Transfer learning from pre-trained transformers",
            "Data augmentation (back-translation, synonym replacement)",
            "Model distillation for faster inference",
            "Ensemble methods for robustness"
        ]
    )

    proposal.define_architecture(
        components={
            "Data Ingestion": "Kafka for streaming data",
            "Model Serving": "TorchServe/TensorFlow Serving",
            "API Gateway": "FastAPI REST API",
            "Dashboard": "Streamlit/Gradio",
            "Database": "PostgreSQL + Redis cache"
        },
        pipeline_steps=[
            "1. Data collection from Twitter API",
            "2. Preprocessing pipeline",
            "3. Model inference",
            "4. Result storage",
            "5. Dashboard visualization"
        ],
        infrastructure="AWS (EC2, RDS, ElastiCache) or GCP"
    )

    proposal.define_evaluation(
        primary_metric="F1-Score (weighted)",
        secondary_metrics=["Accuracy", "Precision", "Recall", "AUC-ROC"],
        evaluation_strategy="5-fold stratified cross-validation + holdout test set",
        target_performance={
            "F1-Score": 0.85,
            "Accuracy": 0.87,
            "Inference Time": "< 50ms"
        }
    )

    proposal.define_deployment(
        deployment_platform="AWS with Docker containers",
        api_type="REST API with FastAPI",
        monitoring_strategy="Prometheus + Grafana for metrics, logging with ELK stack",
        scaling_requirements="Auto-scaling based on CPU/memory, handle 1000 req/s peak"
    )

    proposal.add_risk(
        "Model drift over time",
        "Accuracy degrades as language/topics evolve",
        "Implement continuous monitoring and periodic retraining"
    )
    proposal.add_risk(
        "API rate limits",
        "Limited data collection from social platforms",
        "Implement caching and use multiple API keys"
    )
    proposal.add_risk(
        "Bias in predictions",
        "Unfair treatment of certain demographics",
        "Regular bias audits, diverse training data"
    )

    proposal.add_deliverable("Trained Model", "Fine-tuned DistilBERT with weights")
    proposal.add_deliverable("REST API", "FastAPI application with Docker")
    proposal.add_deliverable("Dashboard", "Interactive Streamlit dashboard")
    proposal.add_deliverable("Documentation", "Technical docs and user guide")
    proposal.add_deliverable("Evaluation Report", "Model performance analysis")

    return proposal


def create_image_classification_proposal():
    """
    Sample: Medical Image Classification Capstone Project
    """
    proposal = CapstoneProjectProposal()

    proposal.set_project_info(
        title="Medical Image Classification for Disease Detection",
        author="ML Student",
        description="""
        Develop a deep learning system for classifying medical images
        (X-rays/CT scans) to assist radiologists in detecting diseases
        with explainable AI features.
        """,
        domain="Computer Vision / Healthcare",
        project_type="Multi-class Image Classification"
    )

    proposal.define_problem(
        problem_statement="""
        Radiologists face high workloads and diagnostic errors can be costly.
        AI assistance can help prioritize cases and provide second opinions.
        """,
        business_value="""
        - Reduce diagnostic time by 40%
        - Decrease false negatives
        - Enable remote screening
        - Support radiologist workflow
        """,
        target_users=["Radiologists", "Healthcare providers", "Hospitals"],
        success_criteria=[
            "Sensitivity > 95% for critical findings",
            "Specificity > 90%",
            "Explainability with attention maps",
            "HIPAA-compliant deployment"
        ]
    )

    proposal.define_dataset(
        name="ChestX-ray14 + NIH Dataset",
        source="NIH Clinical Center / Kaggle",
        size="112,120 X-ray images",
        features=["pixel_data", "patient_age", "patient_gender", "view_position"],
        target="disease_labels (14 classes)",
        preprocessing_steps=[
            "1. Image resizing and normalization",
            "2. Data augmentation (rotation, flip, contrast)",
            "3. Class balancing (oversampling rare diseases)",
            "4. Train/val/test split by patient ID"
        ],
        potential_biases=["Age/gender imbalance", "Equipment variation", "Geographic bias"]
    )

    proposal.define_methodology(
        approach="Supervised Learning with Transfer Learning",
        algorithms=["ResNet-50", "DenseNet-121", "EfficientNet-B4"],
        baseline_model="ResNet-50 pre-trained on ImageNet",
        advanced_model="DenseNet-121 with attention mechanism",
        techniques=[
            "Transfer learning from ImageNet",
            "Grad-CAM for explainability",
            "Multi-label classification",
            "Ensemble of models"
        ]
    )

    proposal.define_evaluation(
        primary_metric="AUC-ROC (per class and mean)",
        secondary_metrics=["Sensitivity", "Specificity", "F1-Score"],
        evaluation_strategy="Patient-level split, 5-fold CV",
        target_performance={
            "Mean AUC-ROC": 0.85,
            "Sensitivity": 0.95,
            "Specificity": 0.90
        }
    )

    proposal.define_deployment(
        deployment_platform="On-premise hospital servers (HIPAA compliant)",
        api_type="REST API with TLS encryption",
        monitoring_strategy="Model performance tracking, drift detection",
        scaling_requirements="Handle 100 images/hour per hospital"
    )

    proposal.add_deliverable("Trained Models", "Ensemble of CNNs with weights")
    proposal.add_deliverable("Explainability Module", "Grad-CAM visualization")
    proposal.add_deliverable("API Service", "Secure REST API")
    proposal.add_deliverable("Clinical Validation Report", "Performance on test cohort")

    return proposal


# ==================== MAIN ====================

def main():
    """
    Demonstrate capstone project planning
    """
    print("=" * 70)
    print("DAY 29: CAPSTONE PROJECT PLANNING")
    print("=" * 70)

    print("""
    This module provides templates and tools for planning your capstone project.

    A good capstone project should:
    1. Solve a real-world problem
    2. Combine multiple skills from the course
    3. Include proper evaluation and documentation
    4. Have a deployment component
    5. Consider ethics and bias

    Let's look at two example proposals:
    """)

    # Create and display sample proposals
    print("\n" + "=" * 70)
    print("SAMPLE PROJECT 1: NLP - Sentiment Analysis")
    print("=" * 70)

    sentiment_proposal = create_sentiment_analysis_proposal()
    sentiment_proposal.display()
    sentiment_proposal.save("sentiment_capstone_proposal.json")

    print("\n" + "=" * 70)
    print("SAMPLE PROJECT 2: Computer Vision - Medical Imaging")
    print("=" * 70)

    medical_proposal = create_image_classification_proposal()
    medical_proposal.display()
    medical_proposal.save("medical_capstone_proposal.json")

    print("\n" + "=" * 70)
    print("YOUR TURN!")
    print("=" * 70)
    print("""
    To create your own proposal:

    1. Create a CapstoneProjectProposal instance
    2. Fill in all sections using the methods:
       - set_project_info()
       - define_problem()
       - define_dataset()
       - define_methodology()
       - define_architecture()
       - define_evaluation()
       - define_deployment()
       - add_risk()
       - add_deliverable()
    3. Save your proposal with save()

    Example:
    ```python
    proposal = CapstoneProjectProposal()
    proposal.set_project_info(
        title="My Awesome Project",
        author="Your Name",
        ...
    )
    proposal.save("my_proposal.json")
    ```
    """)

    return sentiment_proposal, medical_proposal


if __name__ == "__main__":
    proposals = main()
