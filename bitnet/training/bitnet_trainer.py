

from transformers import TrainingArguments
from trl import SFTTrainer
import os
import wandb 

# wandb.login(key = os.environ['WANDB_API_KEY'] )

class BitNetTrainer:
    def __init__(self, output, max_seq_length=512, batch_size=1, epochs=2, learning_rate=2e-4):
        self.output = output
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def train(self, model, dataset):
        wandb.init(project="1bitllm_finetuning", name="expt_001")
        
        tokenizer = model.tokenizer
        model = model.model

        training_args = TrainingArguments(
            output_dir=self.output,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            logging_steps=1,
            num_train_epochs=self.epochs,
            save_steps=500,
            save_total_limit=2,
            # report_to = "wandb"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset.dataset,
            max_seq_length=self.max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
            data_collator=dataset.data_collator
        )

        trainer.train()

        output_dir = os.path.join(self.output, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
