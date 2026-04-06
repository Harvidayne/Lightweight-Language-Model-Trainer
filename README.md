A lightweight framework for training, experimenting with, and serving small language models (SLMs) using the HuggingFace ecosystem.

This project demonstrates how a modern machine learning pipeline is structured — including dataset processing, model training, experiment tracking, and inference APIs — while keeping the codebase modular and easy to understand.

Project Overview

Training language models can be complex. This project provides a minimal yet realistic architecture for building and deploying language models.

The pipeline includes:

Dataset → Preprocessing → Tokenization → Model Training → Experiment Tracking → Inference API

This repository is designed for:

ML engineers learning transformer pipelines
developers experimenting with language models
researchers prototyping small language models
students studying ML system architecture
Key Features
Modular Architecture

The codebase is organized into separate modules for configuration, datasets, models, training, inference, and utilities.

Dataset Pipeline

Supports multiple dataset sources:

Local text files
JSON / JSONL datasets
HuggingFace datasets
Model Training

Fine-tune transformer language models using HuggingFace Trainer.

Experiment Tracking

Training runs automatically log metrics and plots:

experiments/logs/<run_id>/
    metrics.json
    loss.png

Metrics tracked include:

training loss
evaluation loss
learning rate
training epochs
CLI Interface

Unified command-line interface:

python main.py train
python main.py generate
python main.py serve
python main.py dataset
Inference API

Serve the model with a FastAPI server.

Example endpoint:

POST /generate

Example request:

{
 "prompt": "Explain neural networks"
}

Example response:

{
 "generated_text": "Neural networks are computational models inspired by the human brain..."
}
