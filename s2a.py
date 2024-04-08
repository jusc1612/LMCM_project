import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from datasets import load_dataset

os.environ['HF_TOKEN'] = ''
os.environ["REPLICATE_API_TOKEN"] = ""

# ------ Load Data ---------

def load_sttn(model, ent_type, sem=True, return_dataset=False):
    if ent_type == "Multi" and sem:
        file = pd.read_csv(f'../Sorting-Through-The-Noise/data/combined_data/multiple_entity_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

    elif ent_type == "Multi" and not sem:
        file = pd.read_csv(f'../Sorting-Through-The-Noise/data/combined_data/multiple_entity_with_neutral_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")
    
    elif ent_type == "Single" and sem:
        file = pd.read_csv(f'../Sorting-Through-The-Noise/data/combined_data/single_entity_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

    elif ent_type == "Single" and not sem:
        file = pd.read_csv(f'../Sorting-Through-The-Noise/data/combined_data/single_entity_with_neutral_distractor/{model}/complete_data_For_MultipleEntityObjectDistractorAccuracy{model}.csv', sep="\t")

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
        file = pd.read_json('../comps/data/comps/comps_wugs_dist-in-between.jsonl', lines=True)
    if before:
        file = pd.read_json('../comps/data/comps/comps_wugs_dist-before.jsonl', lines=True)
    if base:
        file = pd.read_json('../comps/data/comps/comps_wugs.jsonl', lines=True)
    
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
    
    pd.DataFrame(data).to_csv(f"/local/js/LMCM_project/eval/{file_spec}.csv", index=False)