# Downloadign the data 
#%%
# from datasets import load_dataset
# dataset = load_dataset("mosaicml/instruct-v3")
# # %%
# train_df = dataset['train'].to_pandas()
# sample_train = train_df.sample(n = 500) 
# train_df_new = train_df.drop(index = sample_train.index)
# sample_test = train_df_new.sample(n = 100) 

# sample_train = sample_train.reset_index(drop = True)#.rename(columns = {'prompt': "prompt", 'response': "answer"})
# sample_test = sample_test.reset_index(drop = True)#.rename(columns = {'prompt': "prompt", 'response': "answer"})



# # saving the training and testing data 
# sample_train.to_json("train_data.jsonl",orient='records', lines=True) 
# sample_test.to_json("test_data.jsonl",orient='records', lines=True)
# %%


# from bitnet.training.bitnet_trainer import BitNetTrainer
# from bitnet.data.sft_data_module import SFTDataModule
# from bitnet.models.bitnet import BitNetLLM

# import argparse
# import os
# import json

# def main():
#     # parse command line arguments
#     parser = argparse.ArgumentParser(description='Train a BitNet')
#     parser.add_argument('-m', '--model', required=True, type=str, help='base model to start with')
#     parser.add_argument('-d', '--dataset', required=True, type=str, help='dataset to train model on')
#     parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
#     parser.add_argument('-n', '--num_samples', default=-1, type=int, help='how many examples to train on')
#     parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for training')
#     parser.add_argument('-e', '--epochs', default=2, type=int, help='Number of epochs to train for')
#     parser.add_argument('-l', '--learning_rate', default=2e-4, type=float, help='Learning rate of the model')
#     parser.add_argument('--max_seq_len', default=512, type=int, help='max sequence length for model')
#     args = parser.parse_args()

#     # mkdir output if not exists
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)

#     # instantiate model / tokenizer
#     model = BitNetLLM(args.model)
    
#     # load dataset
#     dataset = SFTDataModule(
#         tokenizer=model.tokenizer,
#         data_path=args.dataset,
#         num_samples=args.num_samples,
#         max_seq_len=args.max_seq_len
#     )
#     print(model.tokenizer.decode(dataset.dataset[0]['input_ids']))
    
#     # save some of the training data for debugging
#     with open(os.path.join(args.output, "debug_data.jsonl"), "w") as f:
#         for i in range(5):
#             data = {
#                 "text": model.tokenizer.decode(dataset.dataset[i]['input_ids']),
#             }
#             f.write(json.dumps(data) + "\n")

#     # kick off the train
#     trainer = BitNetTrainer(args.output, batch_size=args.batch_size, epochs=args.epochs)
#     trainer.train(model, dataset)

# if __name__ == '__main__':
#     main()

#%%
import json 
import pandas as pd 
import torch
# def run_base_model_eval(model, data_path, save_path = "raw_model_preds.csv"): 
#     data = []
#     with open(data_path, "r") as file:

#         for line in file:
#             json_line = json.loads(line.strip())
#             prompt = json_line.get('prompt', '')
#             response = json_line.get('response', '')
#             source = json_line.get('source', '')
#             predict_req = {'prompt': prompt, 'answers': [response], 'source': source}
#             with torch.no_grad():
#                 prediction = model._predict(predict_req)
#             data.append(prediction)
#     df = pd.DataFrame(data)
#     df.to_csv(save_path, index=False)
#     print(df)

from bitnet.training.bitnet_trainer import BitNetTrainer
from bitnet.data.sft_data_module import SFTDataModule
from bitnet.models.bitnet import BitNetLLM
from transformers import TrainingArguments
from trl import SFTTrainer

import os
import json


config = {
    'model': '1bitLLM/bitnet_b1_58-large',
    'dataset': '/kaggle/working/train_data.jsonl',
    'output': './results/bitnet_b1_58-large-instruct',
    'num_samples': -1,  # Use -1 for all samples
    'batch_size': 1,
    'epochs': 2,
    'learning_rate': 2e-4,
    'max_seq_len': 512
}


# instantiate model / tokenizer
model = BitNetLLM(config['model'])
dataset = SFTDataModule(
        tokenizer=model.tokenizer,
        data_path="test_data.jsonl",
        num_samples=-1,
        max_seq_len=512
    )

training_args = TrainingArguments(
        output_dir= "/tmp/dev", 
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=4,
        warmup_steps=30,
        logging_steps=1,
        num_train_epochs=config["epochs"],
        save_steps=500,
        save_total_limit=2,
        # report_to = "wandb"
    )
trainer = SFTTrainer(
    model=model.model,
    eval_dataset=dataset.dataset,
    max_seq_length=config["max_seq_len"],
    tokenizer=model.tokenizer,
    args=training_args,
    dataset_text_field="text",
    data_collator=dataset.data_collator
)
# run_base_model_eval(model, data_path = "test_data.jsonl", save_path = "raw_model_preds.csv")
# %%
results = trainer.evaluate()
# %%
