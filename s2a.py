import os
import replicate
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
from datasets import load_dataset

os.environ['HF_TOKEN'] = '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'
os.environ["REPLICATE_API_TOKEN"] = "r8_7AptjQVz5oOJeZwIUUYNOpzhOeA8nZb32b9ze"

accelerator = Accelerator()

# ------ Load Data ---------

def load_sttn(model, ent_type, sem=True, return_dataset=False):
    if ent_type == "Multi" and sem:
        file = pd.read_csv(f'/local/js/lmcm_project/Sorting-Through-The-Noise/data/combined_data/multiple_entity_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

    elif ent_type == "Multi" and not sem:
        file = pd.read_csv(f'/local/js/lmcm_project/Sorting-Through-The-Noise/data/combined_data/multiple_entity_with_neutral_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")
    
    elif ent_type == "Single" and sem:
        file = pd.read_csv(f'/local/js/lmcm_project/Sorting-Through-The-Noise/data/combined_data/single_entity_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

    elif ent_type == "Single" and not sem:
        file = pd.read_csv(f'/local/js/lmcm_project/Sorting-Through-The-Noise/data/combined_data/single_entity_with_neutral_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

    file['target'] = file['target_occupation']
    del file['target_occupation']

    if return_dataset:
        data = {"sentence": file['sentence'], "target": file['target']}
        return Dataset.from_dict(data)
    
    else:
        return pd.DataFrame({"sentence": file['sentence'], "target": file['target']})

def get_zero_attractors(data1, data2):
    if isinstance(data1, Dataset):
        data1 = pd.DataFrame({'sentence': data1['sentence'], 'target': data1['target']})
        data2 = pd.DataFrame({'sentence': data2['sentence'], 'target': data2['target']})

    concat = pd.concat([data1, data2])
    duplicates = concat[concat.duplicated(subset='sentence', keep=False)]
    unique = duplicates.drop_duplicates(subset='sentence')
    return unique[::4]['sentence'].to_list(), unique[::4]['target'].to_list()

def load_comps(in_between=False, before=False, base=False, return_dataset=False):
    if in_between:
        file = pd.read_json('/local/js/lmcm_project/comps/data/comps/comps_wugs_dist-in-between.jsonl', lines=True)
    if before:
        file = pd.read_json('/local/js/lmcm_project/comps/data/comps/comps_wugs_dist-before.jsonl', lines=True)
    if base:
        file = pd.read_json('/local/js/lmcm_project/comps/data/comps/comps_wugs.jsonl', lines=True)
    
    #sents_acc = [file['prefix_acceptable'][i] + " " + file['property_phrase'][i] for i in range(len(file))]
    #sents_inacc = [file['prefix_unacceptable'][i] + " " + file['property_phrase'][i] for i in range(len(file))]

    sents_acc = [sent + " " + prop for sent, prop in zip(file['prefix_acceptable'], file['property_phrase'])]
    sents_inacc = [sent + " " + prop for sent, prop in zip(file['prefix_unacceptable'], file['property_phrase'])]
    

    data = {'sentence_acc': sents_acc, 
            'sentence_unacc': sents_inacc,
            'acceptable_concept': file['acceptable_concept'], 
            'unacceptable_concept': file['unacceptable_concept'],
            'negative_sample_type': file['negative_sample_type']}

    if return_dataset:
        return Dataset.from_dict(data)
    else:
        return pd.DataFrame(data)

def load_tqa():
    return load_dataset('truthful_qa', 'generation')
    
def load_ic_qa(ic=False, return_dataset=False):
    file = pd.read_csv('/local/js/lmcm_project/factual_qa/qa_ic.csv')

    if ic:
        questions = [file['irrelevant_context'][i] + " " + file['question'][i] for i in range(len(file))]
    else:
        questions = file['question']
    
    data = {'question': questions,
            'true_answer': file['true_answer'],
            'false_answer': file['false_answer']}

    if return_dataset:
        return Dataset.from_dict(data)
    else:
        return pd.DataFrame(data)


# ------- Utils -----------

def preprocess(inputs, tokenizer, model_name, system_prompt, comps=False, prem_hyp=False):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if comps and prem_hyp:
        #inputs = [sent[0].split("Therefore,") for sent in inputs]
        inputs = [[sent_acc.split("Therefore,"), sent_unacc.split("Therefore,")] for sent_acc, sent_unacc in inputs]#zip(inputs[0], inputs[1])]
        #inputs = ["\nPremise: " + " ".join(sent[:-1]) + "\nHypothesis: " + sent[-1].strip().capitalize() + "\nOne-word response (True or False): " for sent in inputs]
        #answers = "\n\nPlease select one of the following options:\nA: Premise 1 \nB: Premise 2 \nC: Uncertain \n\nReply with the single letter corresponding to your choice."
        #inputs = ["\nHypothesis: " + sent[0][-1].strip().capitalize() + "\n\nPlease, select one of the following options: \nA: " + " ".join(sent[1][:-1]) + "\nB: " + " ".join(sent[0][:-1]) + "\nC: Uncertain" + "\n\nOne-letter Response: " for sent in inputs]
        #inputs = ["Hypothesis: " + sent[0][-1].strip().capitalize() +"\nPremise 1: " + " ".join(sent[0][:-1]) + "\nPremise 2: " + " ".join(sent[1][:-1]) + "\n\nRespond by completing the following sentence with one single digit. Only give the digit as your answer!\nAnswer: The hypothesis above logically follows from Premise " for sent in inputs]
        inputs = ["\nA: " + " ".join(sent[1][:-1]) + "\nB: " + " ".join(sent[0][:-1]) + "\nHypothesis: " + sent[0][-1].strip().capitalize() + "\n\nJust generate a single letter 'A' or 'B': " for sent in inputs]

        #print(inputs[0])
    
    if comps and not prem_hyp: 
        # llama: Just respond with 'A' or 'B' as your answer. Answer:
        inputs = ["A: " + sent_1 + "\nB: " + sent_2 + "\n\nJust generate a single letter 'A' or 'B' as your answer:" for sent_1, sent_2 in inputs]

    if 'llama' in model_name:
        def get_template_llama(prompt):
            chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            ]
            return chat

        inputs = [tokenizer.apply_chat_template(get_template_llama(inp), add_generation_prompt=True, tokenize=False) for inp in inputs]
        print(inputs[0])
    
    else:
        def get_template(prompt):
            chat = [
            {"role": "user", "content": system_prompt + "\n" + prompt},
            ]
            return chat

        inputs = [tokenizer.apply_chat_template(get_template(inp), add_generation_prompt=True, tokenize=False) for inp in inputs]
        print(inputs[0])

    model_inputs = tokenizer(inputs, padding=True, truncation=False, pad_to_multiple_of=8)
    dataset = Dataset.from_dict(model_inputs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.with_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)

    return dataset

def write_out(sents, predictions, targets, file_spec, comps=False):
    if comps:
        sents_acc, sents_unacc = sents[0], sents[1]
        if len(predictions) < len(targets):
            length_diff = len(targets) - len(predictions)
            predictions += ['na'] * length_diff
        data = {"sentence_acceptable": sents_acc, "sentence_unacceptable": sents_unacc, "prediction": predictions, "target": targets}
    else:
        if len(predictions) < len(targets):
            length_diff = len(targets) - len(predictions)
            predictions += ['na'] * length_diff
        data = {"sentence": sents, "prediction": predictions,  "target": targets}
    
    pd.DataFrame(data).to_csv(f"/local/js/lmcm_project/eval/{file_spec}.csv", index=False)


# ------ Huggingface Experiments ----------
    
def load_model_acc(model_id, in_4bit=False):

    configuration = LlamaConfig()
    with init_empty_weights():
        model = LlamaModel(configuration)
    
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
    #model = load_and_quantize_model(model, weights_location=weights_loc, bnb_quantization_config=bnb_config, device_map={"": accelerator.process_index})#, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": accelerator.process_index}, token=os.environ['HF_TOKEN'])#, offload_folder="save_folder")

    return model, tokenizer
        
def load_model_hf(model_id, memory_pinning, in_4bit=False, no_quant=False):    
    if no_quant:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side='left') # use_fast argument
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', max_memory=memory_pinning, token=os.environ['HF_TOKEN'])

        return model, tokenizer
    
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side='left') 
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto', max_memory=memory_pinning, token=os.environ['HF_TOKEN'])

    return model, tokenizer

def generate_hf(model, tokenizer, data, temperature=0.6, top_p=0.9, seed=42, N=None, accel=False, save_preds=False):
    set_seed(seed)
    predictions = []

    if accel:
        accelerator.wait_for_everyone()

        with accelerator.split_between_processes(data) as prompts:
            for i, batch in enumerate(tqdm(prompts, desc="Batches")):
                outputs = model.generate(**batch, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id, temperature=temperature, top_p=top_p, do_sample=True)
                
                # remove prompts
                outputs = [output[len(input):] for input, output in zip(batch['input_ids'], outputs)]

                preds = tokenizer.batch_decode(outputs)
                predictions.extend(preds)
            
                if N and i == N:
                    break 
        
        preds = gather_object(preds)


    else:
        for i, batch in enumerate(tqdm(data, desc="Batches")):
            
            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)
            #start.record()

            outputs = model.generate(**batch, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, temperature=temperature, top_p=top_p, do_sample=True)

            '''end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            # check sizes
            throughput = batch['input_ids'].size(0) * len(batch) / latency
            print(throughput)'''
            
            outputs = [output[len(input):] for input, output in zip(batch['input_ids'], outputs)]

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(preds)

            if save_preds:
                file_path = '/local/js/lmcm_project/eval/aa_check_preds.csv'
                try:
                    existing_data = pd.read_csv(file_path)
                except FileNotFoundError:
                    existing_data = pd.DataFrame({"prediction": []})

                new_data = pd.DataFrame({'prediction': preds})
                updated_data = existing_data._append(new_data, ignore_index=True)
                updated_data.to_csv(file_path, index=False)
            
            if isinstance(N, int) and i == N-1:
                break
            # pred = tokenizer.batch_decode(outputs[:, inp_length:], skip_special_tokens=True)[0]
            # response = tokenizer.decode(output[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).strip()
    
    return predictions

# ------ some helpers --------


'''def completion_langchain(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    llm = Replicate(
        model=model,
        model_kwargs={"temperature": temperature,"top_p": top_p, "max_new_tokens": 1000}#, "system_prompt":system_prompt}
    )
    return llm(prompt)'''

'''def complete_and_print_langchain(prompt: str, model: str = DEFAULT_MODEL):
    print(f'==============\n{prompt}\n==============')
    response = completion_langchain(prompt, model)
    print(response, end='\n\n')'''


# ------- Replicate --------------

def run_r_llama_instance(model, prompt, system_prompt, temperature=0.6, top_p=0.9):
    output = replicate.run(
        model,
        input={
            "top_p": top_p,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature
            }
        )

    print(model)
    return output
    #for item in output:
     #   print(item.lower(), end="")

def run_model_replicate(model, data, targets, system_prompt, temperature=0.6, top_p=0.9, out=False, file_spec=None):
    predictions = []
    
    if 'llama' in model:
        for sent in data:
            output = replicate.run(
                model,
                input={
                    "top_p": top_p,
                    "prompt": sent,
                    "system_prompt": system_prompt,
                    "temperature": temperature
                    }
                )
        
            answer = "".join(list(output))
            predictions.append(answer)
    
    if 'mistral' in model:
        for sent in data:
            output = replicate.run(
                model,
                input={
                    "top_k": 50,
                    "top_p": top_p,
                    "prompt": sent,
                    "temperature": temperature,
                    "max_new_tokens": 2048,
                    "prompt_template": "<s>[INST] {prompt} [/INST] ",
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                    }
                )
            
            answer = "".join(list(output))
            predictions.append(answer)

    if 'gemma' in model:
        for sent in data:
            output = replicate.run(
                model,
                input={
                    "top_k": 50,
                    "top_p": top_p,
                    "prompt": sent,
                    "temperature": temperature,
                    "max_new_tokens": 512,
                    "min_new_tokens": -1,
                    "repetition_penalty": 1
                    }
                )

            answer = "".join(list(output))
            predictions.append(answer)

    if out and file_spec:
        write_out(data, predictions, targets, file_spec)

    return predictions

def run_r_mistral(prompt, model, temperature=0.6, top_p=0.9):
    output = replicate.run(
    model,
    input={
        "top_k": 50,
        "top_p": top_p,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": 2048,
        "prompt_template": "<s>[INST] {prompt} [/INST] ",
        "presence_penalty": 0,
        "frequency_penalty": 0
        }
    )

    for item in output:
        print(item, end="")

def run_r_gemma(prompt, model, temperature=0.7, top_p=0.95):
    output = replicate.run(
        model,
        input={
            "top_k": 50,
            "top_p": top_p,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": 512,
            "min_new_tokens": -1,
            "repetition_penalty": 1
            }
        )

    for item in output:
        print(item, end="")

# ------ might be useful ------------

'''for j, output in enumerate(outputs):
    inp_length = batch['input_ids'][j].shape[0]
    pred = tokenizer.decode(output[inp_length:], skip_special_tokens=True)
    predictions.append(pred)'''