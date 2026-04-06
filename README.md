# Lightweight-Language-Model-Trainer

A lightweight framework for training, experimenting with, and serving small language models (SLMs) using the HuggingFace ecosystem.

This project demonstrates how a modern machine learning pipeline is structured — combining dataset processing, model training, experiment tracking, and inference APIs in a simple and modular codebase.

Project Overview

This project provides a minimal but realistic implementation of a language model training system.

Pipeline:

Dataset → Preprocessing → Tokenization → Model Training → Experiment Tracking → Inference API

It is designed for:

learning transformer model training
experimenting with HuggingFace models
building AI-powered applications
understanding ML system architecture
Features
Modular Architecture

The project is organized into clear modules:

config/
datasets/
models/
training/
inference/
experiments/
utils/
Dataset Pipeline

Supports multiple dataset formats:

local text files
JSON / JSONL datasets
HuggingFace datasets
Model Training

Fine-tune transformer models using HuggingFace Trainer.

Experiment Tracking

Training runs log metrics automatically:

experiments/logs/<run_id>/
    metrics.json
    loss.png
CLI Interface

Unified command-line interface.

Your main.py CLI supports commands such as:

python main.py train
python main.py generate
python main.py serve
python main.py dataset

These commands are implemented in the main CLI controller .

Inference API

Run a FastAPI server for model inference.

Example endpoint:

POST /generate

Example request:

{
 "prompt": "Explain neural networks"
}

Example response:

{
 "generated_text": "Neural networks are computational models..."
}
