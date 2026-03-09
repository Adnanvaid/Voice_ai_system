#!/usr/bin/env python
# coding: utf-8

# In[14]:


import whisper
import requests
import time
import json
import os
from jiwer import wer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# In[15]:


class VoiceAIEvaluationPipeline:

    def __init__(self, whisper_model="base", llm_model="llama3", fp16=False):

        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(whisper_model)

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.llm_model = llm_model
        self.ollama_url = "http://host.docker.internal:11434/api/generate"

        self.fp16 = fp16 

    def transcribe_audio(self, audio_path):
        result = self.whisper_model.transcribe(audio_path, fp16=self.fp16)
        return result["text"]

    def query_llm(self, prompt):

        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }

        response = requests.post(self.ollama_url, json=payload)

        return response.json()["response"]

    def compute_wer(self, reference, hypothesis):
        return wer(reference, hypothesis)

    def semantic_similarity(self, text1, text2):

        emb1 = self.embed_model.encode([text1])
        emb2 = self.embed_model.encode([text2])

        similarity = cosine_similarity(emb1, emb2)[0][0]

        return float(similarity)

    def hallucination_rate(self, reference, prediction):

        similarity = self.semantic_similarity(reference, prediction)

        if similarity < 0.5:
            return 1

        return 0

    def evaluate_sample(self, audio_path, ground_truth):

        start_time = time.time()

        transcription = self.transcribe_audio(audio_path)

        llm_response = self.query_llm(transcription)

        latency = time.time() - start_time

        wer_score = self.compute_wer(
            ground_truth["transcript"],
            transcription
        )

        semantic_score = self.semantic_similarity(
            ground_truth["response"],
            llm_response
        )

        hallucination = self.hallucination_rate(
            ground_truth["response"],
            llm_response
        )

        result = {
            "audio_file": audio_path,
            "transcription": transcription,
            "llm_response": llm_response,
            "metrics": {
                "latency": latency,
                "wer": wer_score,
                "semantic_similarity": semantic_score,
                "hallucination_rate": hallucination
            }
        }

        return result


# In[16]:


def run_evaluation(dataset_path):

    pipeline = VoiceAIEvaluationPipeline()

    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []

    for sample in dataset:

        audio = sample["audio"]
        ground_truth = sample["ground_truth"]

        print(f"Evaluating: {audio}")

        result = pipeline.evaluate_sample(audio, ground_truth)

        results.append(result)

    os.makedirs("reports", exist_ok=True)

    output_path = "reports/evaluation_report.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation completed.")
    print("Report saved to:", output_path)


# In[17]:


if __name__ == "__main__":

    dataset_file = "data/ground_truth.json"

    run_evaluation(dataset_file)


# In[ ]:




