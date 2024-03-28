from torch.utils.data import DataLoader
from transformers import default_data_collator
from importlib import reload
import s2a
import argparse
import pandas as pd
import random

s2a = reload(s2a)
random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, required=True, choices=['comps', 'noise', 'tqa', 'qa_ic'], help='Which dataset to use')
    parser.add_argument('--prem_hyp', action="store_true", required=False, help='For COMPS only: Split data into labeled Premise and Hypothesis')
    parser.add_argument('--subset', type=str, required=True, choices=['oracle', 'multiSem', 'multiNeutral', 'singleSem', 'singleNeutral', 'inBetween', 'before', 'ic'], help='Data subset to evaluate')
    parser.add_argument('--subset_size', type=int, required=False, help='Choose subset size to evaluate')
    parser.add_argument('--model', type=str, required=True, choices=['llama2-13', 'llama2-70', 'mistral-v2', 'gemma-7b-it'], help='Model for evaluation')
    parser.add_argument('--quant', type=str, choices=['4bit', '8bit'], help='Set quantization method')
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
    if dataset == 'noise':
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

    if dataset == 'tqa':
        qa_data = s2a.load_qa()
        sents, targets = qa_data['validation']['question'], qa_data['validation']['best_answer']
    
        system_prompt = "Answer the following question: "

    if dataset == 'qa_ic':
        if data_subset == 'oracle':
            qa_data = s2a.load_ic_qa()
        if data_subset == 'ic':
            qa_data = s2a.load_ic_qa(ic=True)

        sents, targets = qa_data['question'], qa_data['true_answer']
        system_prompt = "Answer the following question: "

    # ----- Models -------
    if model_name == 'llama2-13':
        model_id = 'meta-llama/Llama-2-13b-chat-hf'
    if model_name == 'llama2-70':
        model_id = 'meta-llama/Llama-2-70b-chat-hf'
    if model_name == 'mistral-v2':
        model_id = 'mistralai/Mistral-7B-Instruct-v0.2'

    # ------ Prompt templates --------
    # following W&S (2023), baseline, oracle and s2a step 2 share the same system prompt to generate the final answer; the system prompt is loosely based on W&S (2023) Fig. 16

    # s2a step 1 prompt is loosely based on W&S (2023), Fig. 15
    fig_15 = '''Given the following text by a user, extract the part that is related and useful, so that
    using that text alone would be good context for providing an accurate and correct answer
    to the question portion of the text. Please include the actual question or query that the
    user is asking. Separate this into two categories labeled with ”Context text related to the
    question (includes all content except unrelated sentences):” and ”Detailed question:”. Do
    not use list.'''

    # ------ Run model -------
    in_4bit = True if quant == "4bit" else False
    N = num_batches if isinstance(num_batches, int) else None

    max_memory_mapping = {0: "0GB", 1: "30GB", 2: "0GB", 3:"30GB", 4:"0GB", 5:"0GB", 6:"0GB", 7:"0GB"}
    model, tokenizer = s2a.load_model_hf(model_id, memory_pinning=max_memory_mapping, in_4bit=in_4bit)
    #model.to_bettertransformer()

    # step 1 of s2a: separate relevant from distracting information 
    if sys2att:
        if dataset == 'noise':
            label = "Context text related to the incomplete sentence (includes all content except unrelated sentences):"
            s2a_step1 = f'''Given the following incomplete sentence, extract the part that is related and useful, so that 
            using that text alone would a be good context for providing an accurate completion to the incomplete sentence. 
            Start your response with "{label}".
            
            Text by user: '''
        
        if dataset == 'comps':
            label = "Context relevant to conlcusion (includes all content except irrelevant sentences):"
            s2a_step1 = f'''Given the following text by a user, extract the part that is related and useful, so that using that text alone would be sufficient context for stating whether the conclusion is true or false.
            Start your response with "{label}".
            
            Text by user: '''

        if dataset in ['qa', 'qa_ic']:
            label = "Question (does not include irrelevant context):"
            s2a_step1 = f'''Given the following text by a user, remove the the portion that is irrelevant to answer the question. Start your response with "{label}".
            
            Text by User: '''
            
        data = s2a.preprocess(sents, tokenizer, model.name_or_path, s2a_step1)
        file_spec = f"{dataset}_{data_subset}_s2a_step1_{model_name}"
        dataloader = DataLoader(data, collate_fn=default_data_collator, batch_size=batch_size)
        preds = s2a.generate_hf(model, tokenizer, dataloader, sents, targets, N=N, out=out, file_spec=file_spec)
        sents = s2a.post_proc_s2a(preds, dataset, )
        file_spec = f"{dataset}_{data_subset}_s2a_step2_{model_name}"

    else:
        file_spec = f"{dataset}_{data_subset}_baseline_{model_name}"

    # prepare data
    if prem_hyp and dataset == 'comps':
        data = s2a.preprocess(sents, tokenizer, model.name_or_path, system_prompt, comps=True, prem_hyp=True)
        sents = (comps_data['sentence_acc'], comps_data['sentence_unacc'])
    if not prem_hyp and dataset == 'comps':
        data = s2a.preprocess(sents, tokenizer, model.name_or_path, system_prompt, comps=True)
        sents = (comps_data['sentence_acc'], comps_data['sentence_unacc'])
    else:
        data = s2a.preprocess(sents, tokenizer, model.name_or_path, system_prompt)

    #print(tokenizer.convert_ids_to_tokens(data['input_ids'][0], skip_special_tokens=True)) 
    dataloader = DataLoader(data, collate_fn=default_data_collator, batch_size=batch_size)

    # generate predictions
    preds = s2a.generate_hf(model, tokenizer, dataloader, N=N, save_preds=True)
    pd.DataFrame({"prediction": preds}).to_csv(f"/local/js/lmcm_project/eval/save_preds.csv", index=False)

    if out:
        s2a.write_out(sents, preds, targets, file_spec, comps=dataset=='comps')

    if not out and num_batches <= 5:
        print(preds) 

if __name__ == '__main__':
    main()
