from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, default_data_collator
from torch.utils.data import DataLoader
from importlib import reload
import s2a
import argparse
import pandas as pd
import random
import os
import torch
from datasets import Dataset
from tqdm import tqdm

s2a = reload(s2a)
random.seed(42)

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
            {"role": "user", "content": system_prompt + prompt},
            ]
            return chat

        inputs = [tokenizer.apply_chat_template(get_template(inp), add_generation_prompt=True, tokenize=False) for inp in inputs]
        print(inputs[0])

    model_inputs = tokenizer(inputs, padding=True, truncation=False, pad_to_multiple_of=8)
    dataset = Dataset.from_dict(model_inputs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = dataset.with_format(type='torch', columns=['input_ids', 'attention_mask'], device=device)

    return dataset

def load_model_hf(model_id, memory_pinning, in_4bit=False, no_quant=False):    
    if no_quant:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side='left') # use_fast argument
        #model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', max_memory=memory_pinning, token=os.environ['HF_TOKEN'])
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', token=os.environ['HF_TOKEN'])

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

def generate_hf(model, tokenizer, data, temperature=0.6, top_p=0.9, seed=42, N=None, save_preds=False):
    set_seed(seed)
    predictions = []

    for i, batch in enumerate(tqdm(data, desc="Batches")):
        outputs = model.generate(**batch, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, temperature=temperature, top_p=top_p, do_sample=True)
        outputs = [output[len(input):] for input, output in zip(batch['input_ids'], outputs)]

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(preds)

        if save_preds:
            file_path = '/local/js/LMCM_project/eval/aa_check_preds.csv'
            try:
                existing_data = pd.read_csv(file_path)
            except FileNotFoundError:
                existing_data = pd.DataFrame({"prediction": []})

            new_data = pd.DataFrame({'prediction': preds})
            updated_data = existing_data._append(new_data, ignore_index=True)
            updated_data.to_csv(file_path, index=False)
        
        if isinstance(N, int) and i == N-1:
            break
    
    return predictions

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, choices=['comps', 'sttn', 'tqa', 'qa_ic'], help='Which dataset to use')
    parser.add_argument('--prem_hyp', action="store_true", required=False, help='For COMPS only: Split data into labeled Premise and Hypothesis')
    parser.add_argument('--subset', type=str, required=True, choices=['oracle', 'multiSem', 'multiNeutral', 'singleSem', 'singleNeutral', 'inBetween', 'before', 'ic'], help='Data subset to evaluate')
    parser.add_argument('--subset_size', type=int, required=False, help='Choose subset size to evaluate')
    parser.add_argument('--model', type=str, required=True, choices=['llama2-13', 'llama2-70', 'mistral-v2', 'gemma-7b-it'], help='Model for evaluation')
    parser.add_argument('--quant', type=str, choices=['4bit', '8bit', 'none'], help='Set quantization method')
    parser.add_argument('--s2a', action="store_true", required=False, default=False, help='Use s2a or not (baseline)')
    parser.add_argument('--batch_size', type=int, default=8, help='Define batch size')
    parser.add_argument('--eval_output_dir', action="store_true", help='Save the model output to a file')
    parser.add_argument('--num_batches', type=int, default=None, help='Choose number of batches to run model on')
    
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    prem_hyp = args.prem_hyp
    data_subset = args.subset
    subset_size = args.subset_size
    model_name = args.model
    quant = args.quant
    sys2att = args.s2a
    batch_size = args.batch_size
    out = args.eval_output_dir
    num_batches = args.num_batches

    # ----- load data ---------
    if dataset == 'sttn':
        model_data = "GPT2Large"

        if data_subset == 'multiSem':
            multiSem = s2a.load_sttn(model_data, "Multi", return_dataset=True)
            sents, targets = multiSem['sentence'], multiSem['target']
        if data_subset == 'multiNeutral':
            multiNeutral = s2a.load_sttn(model_data, "Multi", sem=False, return_dataset=True)
            sents, targets = multiNeutral['sentence'], multiNeutral['target']
        if data_subset == 'singleSem':
            singleSem =  s2a.load_sttn(model_data, "Single", return_dataset=True)
            sents, targets = singleSem['sentence'], singleSem['target']
        if data_subset == 'singleNeutral':
            singleNeutral = s2a.load_sttn(model_data, "Single", sem=False, return_dataset=True)
            sents, targets = singleNeutral['sentence'], singleNeutral['target']

        # set system prompt
        system_prompt = 'Complete the following sentence with one single word: '
    
    if dataset == 'comps':
        if data_subset == 'oracle':
            comps_data = s2a.load_comps(base=True)
        if data_subset == 'inBetween':
            comps_data = s2a.load_comps(in_between=True)
        if data_subset == 'before':
            comps_data = s2a.load_comps(before=True)
        
        if isinstance(subset_size, int):
            comps_data = comps_data.sample(n=subset_size, random_state=42)
        
        sents = [[acc, unacc] for acc, unacc in zip(comps_data['sentence_acc'], comps_data['sentence_unacc'])] #, (comps_data['acceptable_concept'], comps_data['unacceptable_concept'])

        for pair in sents:
            random.shuffle(pair)
        
        targets = ["A" if rand[0] == acc else "B" for rand, acc in zip(sents, comps_data['sentence_acc'])]

        # set system prompt
        #system_prompt = "State whether the conclusion can be deducted from the premises. Separate into two categories labeled with 'Reasoning:' and 'Final answer (true or false):' \nSentence: "
        #system_prompt = f"True or false. Give a single-word answer, please. \nExample: \nStatement: {sents[0]} Conclusion: True"
        #system_prompt = "Is the following statement true or false? Just answer with 'True' or 'False'. Your answer should not depend on potential nonce words. \nStatement: "
        #system_prompt = "Given the premise and hypothesis provided, does the hypothesis logically follow from the premise? Give a single-word response: Respond with 'True' if the hypothesis can be inferred from the premise, or 'False' if it cannot."
        system_prompt = "Given the premise and hypothesis provided, simply respond with 'True' if the hypothesis can be logically inferred from the premise, or 'False' if it cannot."
        system_prompt = "Given the following premises, please determine for which one the conclusion can be drawn from:\n\n"
        system_prompt = "You are provided with a hypothesis and two premises labeled 'A' and 'B'. Select the premise from which the hypothesis logically follows. Select 'C' if you're uncertain. Reply with the single letter corresponding to your choice."
        system_prompt = "Select the premise 'A' or 'B' from which the provided hypothesis logically follows. Select 'C' if you're uncertain. Reply with the single letter corresponding to your choice."
        system_prompt = "You'ra a helpful writing assistant. Tell me from which premise (1 or 2) the given hypothesis logically follows." # Reply with the single digit corresponding to your choice."
        system_prompt = "You are a helpful writing assistant. Tell me which sentence 'A' or 'B' is semantically more plausible. Reply with the single letter corresponding to your choice."

    # ----- Models -------
    if model_name == 'llama2-13':
        model_id = 'meta-llama/Llama-2-13b-chat-hf'
    if model_name == 'llama2-70':
        model_id = 'meta-llama/Llama-2-70b-chat-hf'
    if model_name == 'mistral-v2':
        model_id = 'mistralai/Mistral-7B-Instruct-v0.2'

    # ------ Run model -------
    if quant in ['4bit', '8bit']:
        in_4bit = True if quant == "4bit" else False
        no_quant=False
    if quant == 'none':
        no_quant = True
        in_4bit = False

    N = num_batches if isinstance(num_batches, int) else None

    max_memory_mapping = {0: "0GB", 1: "0GB", 2: "0GB", 3:"60GB", 4:"0GB", 5:"0GB", 6:"0GB", 7:"0GB"}
    model, tokenizer = load_model_hf(model_id, memory_pinning=max_memory_mapping, in_4bit=in_4bit, no_quant=no_quant)
    #model.to_bettertransformer()

    # step 1 of s2a: separate relevant from distracting information 
    if sys2att:
        if dataset == 'sttn':
            label = "Sentence without irrelvant context:"
            s2a_step1 = f'''Given the following incomplete sentence, extract the part that is related and useful, so that using that part alone
                        would a be good context for providing an accurate completion to the incomplete sentence. Please, start your response
                        with: "{label}" followed by your extracted part.
                        Sentence: '''

            df = pd.read_csv("/local/js/LMCM_project/error_files/test2_multiSem_isIn_mistral.csv")
            sents, targets = df['sentence'], df['target']
        
        if dataset == 'comps':
            label = "Context relevant to conlcusion (includes all content except irrelevant sentences):"
            s2a_step1 = f'''Given the following text by a user, extract the part that is related and useful, so that using that text alone would be sufficient context for stating whether the conclusion is true or false.
            Start your response with "{label}".
            
            Text by user: '''

        if dataset in ['qa', 'qa_ic']:
            label = "Question (does not include irrelevant context):"
            s2a_step1 = f'''Given the following text by a user, remove the the portion that is irrelevant to answer the question. Start your response with "{label}".
            
            Text by User: '''
            
        data = preprocess(sents, tokenizer, model.name_or_path, s2a_step1)
        file_spec = f"{dataset}_{data_subset}_s2a_step1_{model_name}"
        dataloader = DataLoader(data, collate_fn=default_data_collator, batch_size=batch_size)
        preds = generate_hf(model, tokenizer, dataloader, N=N, save_preds=True)
        pd.DataFrame({"sentence": sents[:num_batches*batch_size], "prediction": preds}).to_csv(f"/local/js/LMCM_project/eval/save_preds.csv", index=False)
        
        #sents = s2a.post_proc_s2a(preds, dataset, )
        #file_spec = f"{dataset}_{data_subset}_s2a_step2_{model_name}"

    else:
        file_spec = f"{dataset}_{data_subset}_baseline_{model_name}"

    # prepare data
    if prem_hyp and dataset == 'comps':
        data = preprocess(sents, tokenizer, model.name_or_path, system_prompt, comps=True, prem_hyp=True)
        sents = (comps_data['sentence_acc'], comps_data['sentence_unacc'])
    if not prem_hyp and dataset == 'comps':
        data = preprocess(sents, tokenizer, model.name_or_path, system_prompt, comps=True)
        sents = (comps_data['sentence_acc'], comps_data['sentence_unacc'])
    else:
        data = preprocess(sents, tokenizer, model.name_or_path, system_prompt)

    #print(tokenizer.convert_ids_to_tokens(data['input_ids'][0], skip_special_tokens=True)) 
    dataloader = DataLoader(data, collate_fn=default_data_collator, batch_size=batch_size)

    # generate predictions
    preds = generate_hf(model, tokenizer, dataloader, N=N, save_preds=True)
    pd.DataFrame({"prediction": preds}).to_csv(f"/local/js/LMCM_project/temp_eval/final_save_preds.csv", index=False)

    if out:
        s2a.write_out(sents, preds, targets, file_spec, comps=dataset=='comps')

    if not out and num_batches <= 5:
        print(preds) 

if __name__ == '__main__':
    main()
