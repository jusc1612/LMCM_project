from langchain.llms import Replicate
import os
import torch
import pandas as pd
import string
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from datasets import Dataset
from tqdm import tqdm

from transformers import LlamaModel, LlamaConfig
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import gather_object, load_and_quantize_model, BnbQuantizationConfig
from huggingface_hub import snapshot_download

os.environ['HF_TOKEN'] = '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'
os.environ["REPLICATE_API_TOKEN"] = "r8_7AptjQVz5oOJeZwIUUYNOpzhOeA8nZb32b9ze"

accelerator = Accelerator()

def load_model_acc(model_id, in_4bit=False):

    '''configuration = LlamaConfig()
    with init_empty_weights():
        model = LlamaModel(configuration)'''
    
    #print(f"Number parameters: {model.num_parameters()}")
    #print(f"Memory: {model.get_memory_footprint()}")
    #print(model.hf_device_map)

    # change path    
    weights_loc = snapshot_download(repo_id="meta-llama/Llama-2-70b-chat-hf")
    
    if in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )


    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side='left') # use_fast argument
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": accelerator.process_index}, token=os.environ['HF_TOKEN'])#, offload_folder="save_folder")
    #model = load_and_quantize_model(model, weights_location=weights_loc, bnb_quantization_config=bnb_config, device_map={"": accelerator.process_index})#, token=os.environ['HF_TOKEN'])

    return model, tokenizer

LLAMA2_70B_CHAT_HF = 'meta-llama/Llama-2-70b-chat-hf'
MISTRAL_INSTRUCT_V02_HF = 'mistralai/Mistral-7B-Instruct-v0.2' 
max_memory_mapping = {0: "0GB", 1: "0GB", 2: "0GB", 3:"36GB", 4:"36GB", 5:"36GB", 6:"36GB", 7:"0GB"}
model, tokenizer = load_model_acc(MISTRAL_INSTRUCT_V02_HF, in_4bit=True)

print(f"Number parameters: {model.num_parameters()}")
print(f"Memory: {model.get_memory_footprint()}")
print(model.hf_device_map)
print(model.config)
