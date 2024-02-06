from typing import List
import os
from pathlib import Path
import transformers
import mlflow
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __init__(self, eos_sequence: List[int]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


if Path(".env").is_file():
    load_dotenv(".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
HF_TOKEN = os.getenv("HF_TOKEN")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")

model_name = "meta-llama/Llama-2-13b-chat-hf"
model_task = "text-generation"

# Quantization config
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# HF model config
model_config = transformers.AutoConfig.from_pretrained(model_name, token=HF_TOKEN)

# Load model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    cache_dir=TRANSFORMERS_CACHE,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    cache_dir=TRANSFORMERS_CACHE,
)

# List of stop words
stop_list = [
    "\nCandidate:",
]
stop_list = [tokenizer(w)["input_ids"] for w in stop_list]


generation_pipeline = transformers.pipeline(
    model=model,
    task=model_task,
    tokenizer=tokenizer,
    device_map="auto",
    stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_list)]),
    return_full_text=True,
)

input_example = ["prompt 1", "prompt 2", "prompt 3"]

params = {
    "temperature": 0.75,
    "do_sample": True,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(generation_pipeline, input_example),
    params,
)

mlflow.set_experiment("Transformers Introduction")

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="text_generator",
        input_example=input_example,
        signature=signature,
    )
    
sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = sentence_generator.predict(
    data=[
        "I can't decide whether to go hiking or kayaking this weekend. Can you help me decide?",
    ],
)

for i, formatted_text in enumerate(predictions):
    print(f"Response to prompt {i+1}:\n{formatted_text}\n")
