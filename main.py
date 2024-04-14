# Downloadign the data 
#%%

########## UNCOMMENT FOR DOWNLOADING THE DATA ############
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

#%%
import json 
import pandas as pd 
import torch
from tqdm import tqdm 
from bitnet.training.bitnet_trainer import BitNetTrainer
from bitnet.data.sft_data_module import SFTDataModule
from bitnet.models.bitnet import BitNetLLM
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import json


config = {
    'model': '1bitLLM/bitnet_b1_58-large',
    'num_samples': -1,  # Use -1 for all samples
    'batch_size': 1,
    'epochs': 2,
    'learning_rate': 2e-4,
    'max_seq_len': 512
}


# instantiate model / tokenizer
model = BitNetLLM("/teamspace/studios/this_studio/results/bitnet_b1_58-large-instruct/final_checkpoint")
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
# %%


# CALL TRAINER.TRAIN FOR TRAINING THE MODEL
########## UNCOMMENT FOR GETTING THE PREDICTION FROM MODEL ############
# outputs = trainer.predict(dataset.dataset, predict_with_generate=True)


############ PARUING OUTPUTS FROM MODEL ############
import pickle
import pandas as pd
import numpy as np

def process_outputs(data_path, test_data_path, output_csv_path):
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
    
    test_data = pd.read_json(test_data_path, lines=True)
    final_output_data = []
    
    for pred in range(100):
        output_data = {}
        label_ids = data['label_ids'][pred]
        filtered_label_ids = [id for id in label_ids if id != -100]
        decoded_data = tokenizer.decode(filtered_label_ids)
        prompt = decoded_data.split("Bessie:")[0]
        guess = decoded_data.split("Bessie:")[-1]
        output_data['metrics'] = data['metrics']
        output_data['guess'] = guess
        
        instruction = prompt.split("User:\n")[1][:140]
        try:
            t_data_comp = test_data[test_data['prompt'].str.contains(instruction)]
            output_data['prompt'] = t_data_comp['prompt'].values
            output_data["response"] = t_data_comp['response'].values
        except:
            try:
                instruction = prompt.split("User:\n")[1][:150]
                t_data_comp = test_data[test_data['prompt'].str.contains(instruction)]
                output_data['prompt'] = t_data_comp['prompt']
                output_data["response"] = t_data_comp['response']
            except: 
                print('Error processing data for instruction:', instruction)

        final_output_data.append(output_data)

    outputs_df = pd.DataFrame(final_output_data)
    outputs_df.to_csv(output_csv_path, index=False)

tokenizer = model.tokenizer  # Assuming 'model' and 'tokenizer' are defined earlier in the script.
process_outputs("old_outputs.pickle", "test_data.jsonl", "old_outputs_eval.csv")
process_outputs("new_outputs.pickle", "test_data.jsonl", "new_outputs_eval.csv")

# %%
################ EVALUATING THE MODEL ################
import dspy
import os 
openai = dspy.OpenAI(api_key = os.environ['OPENAI_API_KEY'])
dspy.settings.configure(lm = openai ) 
class FactJudge(dspy.Signature):
    """Judge if the answer is factually similar to the actual answer."""

    ground_truth = dspy.InputField(desc="Ground truth answer of question")
    answer = dspy.InputField(desc="Answer predicted by LLM model")
    factually_correct = dspy.OutputField(desc="Is the answer factually correct in consideration to the actual answer?", prefix="Facual[Yes/No]:")

judge = dspy.ChainOfThought(FactJudge)

def factuality_metric(example):
    factual = judge(ground_truth=example.ground_truth, answer=example.answer)
    return factual.factually_correct=="Yes"
# %%

# evaluating old predictions 
import pandas as pd 
#%%
from tqdm import tqdm 

#%%
def evaluate_predictions(csv_file_path): 
    "evaluates the predictions of model by using LLM as a Judge"
    dataframe_to_evaluate = pd.read_csv(csv_file_path)
    dataframe_to_evaluate['is_correct']  = None 
    for index, row in tqdm(dataframe_to_evaluate.iterrows()):
        example = dspy.Example(ground_truth=row['response'], answer=row['guess'])
        dataframe_to_evaluate.loc[index, 'is_correct'] = factuality_metric(example)   
    dataframe_to_evaluate.to_csv(csv_file_path, index = False)
    print("==========================")
    print("Model correctness ")
    print(dataframe_to_evaluate['is_correct'].value_counts())
    print('---------------------------')
    print("Model metric Accuracy")
    print(dataframe_to_evaluate['is_correct'].value_counts(normalize = True))
    print("===========================")
    return dataframe_to_evaluate
# %%
# evaluate_predictions("./new_outputs_eval.csv")
# %%
import pandas as pd 
df = pd.read_csv("./old_outputs_eval.csv")
print("============OLD OUTPUTS==============")
print("Model correctness ")
print(df['is_correct'].value_counts())
print('---------------------------')
print("Model metric Accuracy")
print(df['is_correct'].value_counts(normalize = True))
print("===========================")

print()

df = pd.read_csv("./new_outputs_eval.csv")
print("============NEW OUTPUTS==============")
print("Model correctness ")
print(df['is_correct'].value_counts())
print('---------------------------')
print("Model metric Accuracy")
print(df['is_correct'].value_counts(normalize = True))
print("===========================")
# %%

################# RESULTS ################
# ===========OLD OUTPUTS==============
# Model correctness 
# is_correct
# True     73
# False    27
# Name: count, dtype: int64
# ---------------------------
# Model metric Accuracy
# is_correct
# True     0.73
# False    0.27
# Name: proportion, dtype: float64
# ===========================

# ============NEW OUTPUTS==============
# Model correctness 
# is_correct
# True     75
# False    25
# Name: count, dtype: int64
# ---------------------------
# Model metric Accuracy
# is_correct
# True     0.75
# False    0.25
# Name: proportion, dtype: float64
# ===========================