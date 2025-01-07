import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
import torch
torch.set_num_threads(1)  # Limit to single thread for CPU operations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

app = FastAPI(title="Sentiment Classification API")

class TextInput(BaseModel):
    text: str

device = torch.device("cpu")

try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    print("Zero-shot classification pipeline loaded successfully!")
except Exception as e:
    print(f"Error initializing the zero-shot classification pipeline: {e}")
    exit(1)

def load_model_and_tokenizer(model_name, state_dict_path=None, peft_adapter_dir=None):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            load_in_4bit=False
        )

        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print("State dictionary loaded successfully.")

        if peft_adapter_dir:
            peft_config = PeftConfig.from_pretrained(peft_adapter_dir)
            model = PeftModel.from_pretrained(model, peft_adapter_dir)
            print("PEFT adapter loaded successfully.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        return model.to('cpu'), tokenizer

    except Exception as e:
        print(f"Error loading model/tokenizer: {str(e)}")
        raise


model_name = "RichardErkhov/meta-llama_-_Llama-3.2-3B-Instruct-8bits"
state_dict_path = "./model/checkpoint-300/rng_state.pth"
peft_adapter_dir = "./model/checkpoint-300/"

model, tokenizer = load_model_and_tokenizer(
    model_name=model_name,
    state_dict_path=state_dict_path,
    peft_adapter_dir=peft_adapter_dir
)

@app.post("/analyze/")
async def analyze_sentiment(text_input: TextInput):
    try:
        model.eval()

        zero_shot_result = classifier(
            text_input.text, 
            candidate_labels=["positive", "negative", "neutral", "irrelevant"]
        )
        zero_shot_label = zero_shot_result["labels"][0]

        item = {
            "instruction": "Detect the sentiment of the input.",
            "input": text_input.text,
            "output": ""
        }
        
        inputs = tokenizer(
            f"### Instruction:\n{item['instruction']}\n### Input:\n{item['input']}\n### Response:\n", 
            return_tensors="pt",
            truncation=True, 
            padding=True
        )

        with torch.no_grad():
            outputs = model.generate(**inputs)
        llama_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        llama_label = "irrelevant"
        for label in ["positive", "negative", "neutral", "irrelevant"]:
            if label in llama_response.lower():
                llama_label = label
                break

        return {
            "zero_shot_sentiment": zero_shot_label,
            "llama_sentiment": llama_label,
            "zero_shot_details": zero_shot_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during sentiment analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
