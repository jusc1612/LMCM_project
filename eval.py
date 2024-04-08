import argparse
import pandas as pd
import string
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from collections import Counter

def post_proc_s2a(preds, targets, label):
    extracted = []
    for pred in preds:
        try:
            assert label in pred, "Cannot find extracted relevant part."
            extracted.append(pred.split(label)[-1].strip())
        except AssertionError as e:
            print('Assertion Error:', e)

    for i, (sent, target) in enumerate(zip(extracted, targets)):
        if target == "meat" and sent.split()[-2] != "sells":
            suffix = f" For his job, {sent.split()[0]} sells"
            extracted[i] = extracted[i] + suffix
        
        if target in ["touchdown", "run"] and not sent.endswith("scored a"):
            suffix = f" In his game, {sent.split()[0]} scored a"
            extracted[i] = extracted[i] + suffix

        if target in ["Helsinki", "Warsaw"] and not sent.endswith("country is"):
            suffix = f" The capital of {sent.split()[0]}'s country is"
            extracted[i] = extracted[i] + suffix
        
        if target in ["Italy", "France", "Peru", "Russia"] and not sent.endswith("traveled to was"):
            suffix = f" the country {sent.split()[0]} traveled to was"
            extracted[i] = extracted[i] + suffix


    return extracted

def post_proc(preds):
    # remove punctuation and lower string
    preds = [pred.translate(str.maketrans('', '', string.punctuation)).lower() for pred in preds]

    # remove >=double whitespaces 
    preds = [" ".join(pred.split()) for pred in preds] 

    return preds

def accuracy(sents, preds, targets, exact_match=False, first_word=False, isin=False, comps=False, errors=False, extension=None):
    correct = 0
    incorrect = {'sentence':[], 'prediction': [], 'target':[]}
    total = len(preds)
    target_acc = []

    if isin:
        for sent, pred, target in zip(sents, preds, targets):
            if target.lower() in pred.lower().split():
                target_idx = pred.index(target.lower())
                if pred[target_idx-1] != "not":
                    correct +=1
                    target_acc.append(target)
            else:
                incorrect['sentence'].append(sent)
                incorrect['prediction'].append(pred)
                incorrect['target'].append(target)
        if errors:
            prefix = "/local/js/LMCM_project/error_files"
            pd.DataFrame(incorrect).to_csv(f"{prefix}/errors_isin_{extension}.csv", index=False)

    if exact_match:
        for sent, pred, target in zip(sents, preds, targets):
            if pred.lower() == target.lower():
                correct +=1
                target_acc.append(target)   
            else:
                incorrect['sentence'].append(sent)
                incorrect['prediction'].append(pred)
                incorrect['target'].append(target)

        if errors:
            prefix = "/local/js/LMCM_project/error_files"
            pd.DataFrame(incorrect).to_csv(f"{prefix}/errors_em_{extension}.csv", index=False)
            
    if first_word:
        for sent, pred, target in zip(sents, preds, targets):
            if pred.split()[0].lower() == target.lower():
                correct +=1                   
                target_acc.append(target)
            else:
                incorrect['sentence'].append(sent)
                incorrect['prediction'].append(pred)
                incorrect['target'].append(target)

        if errors:
            prefix = "/local/js/LMCM_project/error_files"
            pd.DataFrame(incorrect).to_csv(f"{prefix}/errors_fw_{extension}.csv", index=False)
    
    if comps:
        counts = defaultdict(int)
        preds = [pred.strip().split()[0] for pred in preds]
        print(Counter(preds))
        #preds_targets = [(pred, target) for pred, target in zip(preds, targets) if pred in ["A", "B"]]
        #preds, targets = preds_targets[0], preds_targets[1]
        #print(Counter(preds))

        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        print((tn, fp, fn, tp))
        for pred, target in zip(preds, targets):
            if pred == target:
                correct += 1
                counts[f"{target} true"] += 1
            elif pred in ['A', 'B']:
                counts[f"{target} false"] += 1
        
        return round(correct/total, 2), counts

    return round(correct/total, 2), target_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['comps', 'sttn'], help='Which dataset to evaluate')
    parser.add_argument('--subset', type=str, required=False, choices=['oracle', 'oracle_baseline', 'multiSem', 'multiNeutral', 'singleSem', 'singleNeutral', 'inBetween', 'before'], help='Data subset to evaluate')
    parser.add_argument('--model', type=str, required=False, choices=['llama2-13', 'llama2-70', 'mistral-v2', 'gemma-7b-it'], help='Model for evaluation')
    parser.add_argument('--file_name', type=str, required=False)
    parser.add_argument('--num_classes', action="store_true", required=False, help='COMPS only: select whether you want to return the number of predictions per class')
    parser.add_argument('--write_errors', action="store_true", required=False, help='Whether wrong answers should be saved to new file')
    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    subset = args.subset
    model = args.model
    file_name = args.file_name
    num_classes = args.num_classes
    write_err = args.write_errors

    if isinstance(file_name, str):
        file_path = f"eval/{file_name}.csv"
    else:
        file_name = f"{dataset}_{subset}_{model}"
        file_path = f"eval/{file_name}.csv"
    
    data = pd.read_csv(file_path)

    if dataset == "sttn":
        sents, preds, targets = data['sentence'], data['prediction'].to_list(), data['target'].to_list()
    
    if dataset == "comps":
        sents_acc, sents_unacc, preds, targets = data['sentence_acceptable'], data['sentence_unacceptable'], data['prediction'].to_list(), data['target'].to_list()

    if preds[-1] == 'na':
        cutoff = 0
        for i, pred in enumerate(preds):
            if pred == "na":
                cutoff = i
                break
        preds = preds[:cutoff]
        targets = targets[:cutoff]

    if dataset == 'sttn':
        preds = post_proc(preds)
        acc_isin, target_acc_isin = accuracy(sents, preds, targets, isin=True, errors=write_err, extension=file_name)
        acc_em, target_acc_em = accuracy(sents, preds, targets, exact_match=True, errors=write_err, extension=file_name)
        acc_fw, target_acc_fw = accuracy(sents, preds, targets, first_word=True, errors=write_err, extension=file_name)
        
        print(f"Target Distribution: {Counter(targets)}")
        print(f"Target correct Isin: {Counter(target_acc_isin)}")
        print(f"Target correct FW: {Counter(target_acc_fw)}")
        print(f"Target correct EM: {Counter(target_acc_em)}")
        print("--------------------------------------------")
        print(f"Is-in Accuracy:\t\t{acc_isin} \nExact Match Accuracy:\t{acc_em} \nFirst Word Accuracy:\t{acc_fw}")
    
    if dataset == 'comps':
        acc, counts = accuracy(sents_acc, preds, targets, comps=True)
        if num_classes:
            print(f"\t\tA\t|\tB")
            print(f"Actual N\t{targets.count('A')}\t|\t{targets.count('B')}")
            print(f"Pred. true\t{counts['A true']}\t|\t{counts['B true']}")
            print(f"Pred. false\t{counts['A false']}\t|\t{counts['B false']}")
            print("-------------------------------------")
        
        #print(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {recall}")
    
if __name__ == '__main__':
    main()