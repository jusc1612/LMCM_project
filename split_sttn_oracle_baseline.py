import s2a
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['llama2-13', 'llama2-70', 'mistral-v2'])
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True, choices=['multiSem', 'multiNeutral', 'singleSem', 'singleNeutral'])

    args = parser.parse_args()
    print(args)
    
    model = args.model
    file_path = args.file_path
    subset = args.subset

    data = pd.read_csv(file_path)
    singleSem =  s2a.load_sttn("GPT2Large", "Single")
    multiSem =  s2a.load_sttn("GPT2Large", "Multi") 
    oracle_sents, _ = s2a.get_zero_attractors(multiSem, singleSem) 

    oracle = {"sentence": [], "prediction": [], "target": []}
    drop_index = []
    for index, row in data.iterrows():
        if row["sentence"] in oracle_sents:
            drop_index.append(index)
            oracle['sentence'].append(row['sentence'])
            oracle['prediction'].append(row['prediction'])
            oracle['target'].append(row['target'])

    baseline = data.drop(index=drop_index)
    oracle = pd.DataFrame(oracle)

    oracle.to_csv(f"eval/a_sttn_{subset}_oracle_{model}.csv", index=False)
    baseline.to_csv(f"eval/a_sttn_{subset}_baseline_{model}.csv", index=False)

if __name__ == '__main__':
    main()
