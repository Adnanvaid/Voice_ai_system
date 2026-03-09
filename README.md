# Voice_ai_system

Speech-to-LLM Evaluation Pipeline

An end-to-end pipeline that converts speech input into text, generates responses using a local Large Language Model, and evaluates the output using multiple NLP metrics.

The project integrates speech recognition with a locally hosted LLM to demonstrate speech-driven AI systems.

Technologies used include:

OpenAI Whisper

Ollama

Llama 3

Docker

Sentence Transformers


# System Architecture

Pipeline flow:

Audio Input

     ↓
Speech Recognition (Whisper)

     ↓
Transcribed Text

     ↓
Prompt to LLM (Llama3 via Ollama)

     ↓
Generated Response

     ↓
Evaluation Metrics

     ↓
JSON Results Output


# valuation Metrics

The system evaluates model performance using the following metrics:

Word Error Rate (WER)

Measures speech transcription accuracy by comparing predicted text with the reference transcript.

Semantic Similarity

Measures meaning similarity between the generated response and the expected answer using sentence embeddings.

Hallucination Rate

Detects whether the LLM generated unsupported or incorrect information.

Latency

Measures the total response time of the pipeline.


# Running the Project

Install dependencies:
pip install -r requirements.txt

Install Ollama
Download and install:
https://ollama.com/download

Pull the LLM model
ollama pull llama3

Run the pipeline
python main.py


# Key Features

Speech-to-text processing

Local LLM inference

REST API integration

Automated evaluation metrics

Docker containerization

JSON-based results logging