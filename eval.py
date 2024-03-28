import argparse
import pandas as pd
import string

def post_proc_s2a(preds, data_name):
    extracted = []
    if data_name == "noise":
        label = ""
    if data_name == "comps":
        label = ""
    if data_name in ["qa", "qa_ic"]:
        label = ""

    for pred in preds:
        try:
            assert label in pred, "Cannot find extracted relevant part."
            extracted.append(pred.split(label)[-1].strip())
        except AssertionError as e:
            print('Assertion Error:', e)
    
    return extracted

def post_proc(preds):
    # remove punctuation and lower string
    preds = [pred.translate(str.maketrans('', '', string.punctuation)).lower() for pred in preds]

    # remove >=double whitespaces 
    preds = [" ".join(pred.split()) for pred in preds] 

    return preds

def accuracy(preds, targets, exact_match=False, first_word=False, isin=False, comps=False, acceptable=True):
    correct = 0
    total = len(preds)

    if isin:
        for pred, target in zip(preds, targets):
            if target.lower() in pred.lower().split():
                correct +=1
    
    elif exact_match:
        for pred, target in zip(preds, targets):
            if pred.lower() == target.lower():
                correct +=1
    
    elif first_word:
        for pred, target in zip(preds, targets):
            if pred.split()[0].lower() == target.lower():
                correct +=1
    
    elif comps:
        for pred, target in zip(preds, targets):
            '''split_string = 'Final answer (true or false):'
            assert split_string in pred

            pred = pred.split(split_string)[-1].strip()'''
            if target in pred.split():
                correct += 1

    return round(correct/total, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['comps', 'noise'], help='Which dataset to evaluate')
    parser.add_argument('--subset', type=str, required=False, choices=['oracle', 'oracle_baseline', 'multiSem', 'multiNeutral', 'singleSem', 'singleNeutral', 'inBetween', 'before'], help='Data subset to evaluate')
    parser.add_argument('--model', type=str, required=False, choices=['llama2-13', 'llama2-70', 'mistral-v2', 'gemma-7b-it'], help='Model for evaluation')
    parser.add_argument('--abs_path', type=str, required=False)

    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    subset = args.subset
    model = args.model
    file_path = args.abs_path

    if not isinstance(file_path, str):
        file_path = f"eval/{dataset}_{subset}_{model}.csv"
    
    data = pd.read_csv(file_path)

    preds, targets = data['prediction'].to_list(), data['target'].to_list()

    if preds[-1] == 'na':
        cutoff = 0
        for i, pred in enumerate(preds):
            if pred == "na":
                cutoff = i
                break
        preds = preds[:cutoff]
        targets = targets[:cutoff]

    if dataset in ['noise', 'qa', 'qa_ic']:
        preds = post_proc(preds)
        acc_isin = accuracy(preds, targets, isin=True)
        acc_em = accuracy(preds, targets, exact_match=True)
        acc_fw = accuracy(preds, targets, first_word=True)
        print(f"Is-in Accuracy:\t\t{acc_isin} \nExact Match Accuracy:\t{acc_em} \nFirst Word Accuracy:\t{acc_fw}")
    
    if dataset == 'comps':
        acc = accuracy(preds, targets, comps=True)        
        
    print(f"Accuracy:\t{acc}")

if __name__ == '__main__':
    main()