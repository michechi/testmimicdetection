# Callback utils for MIMIC training
from lightning.pytorch.callbacks import Callback
import pandas as pd
class GenerateText(Callback):
    """
    Once in a while generate from the model to see if its working "fine"
    This class is a custom callback for PyTorch Lightning that generates text from a model at the end of each validation epoch and logs it.
    """
    def __init__(self, dataloaders, max_token_len, num_samples=4):
        # Collect some examples per dataset
        self.max_token_len = max_token_len
        self.dataloaders = dataloaders
        self.num_samples=num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        generated_table = []
        outcome_feature = self.dataloaders['validation'].dataset.outcome_feature
        for batch in self.dataloaders['validation']:
                try:
                    data_row = {key : batch[key][0] for key in [outcome_feature]}
                    
                    def generate_from_prompt(prompt_ids, attention_mask_ids, max_new_tokens=20):
                        generated = pl_module.model.generate(input_ids = prompt_ids, attention_mask=attention_mask_ids, max_new_tokens=max_new_tokens, do_sample=False) #True, temperature=0.1, num_beams=3
                        generated = pl_module.tokenizer.decode(generated[0])
                        prompt = pl_module.tokenizer.batch_decode(prompt_ids)[0]
                        prompt = prompt.replace(f"{pl_module.tokenizer.bos_token} ","")
                        generated = generated.replace(prompt,"")
                        return prompt, generated
                    
                    prompt_ids = batch['prompt.input_ids'][:1,]
                    attention_mask_ids = batch['prompt.attention_mask'][:1]
                    prompt, generated = generate_from_prompt(prompt_ids, attention_mask_ids)
                    data_row['prompt'] = prompt
                    data_row['generated']  = generated

                    pstop= 100
                    prompt_ids = batch['prompt.input_ids'][:1,:pstop]
                    attention_mask_ids = batch['prompt.attention_mask'][:1,:pstop]
                    prompt, generated = generate_from_prompt(prompt_ids, attention_mask_ids, max_new_tokens=100)
                    data_row['generated_from_100']  = generated
                    # Dont keep all features
                    
                    data_row = {key: data_row[key] for key in ['prompt','death_status','generated','generated_from_100']}
                    generated_table.append(data_row)
                except Exception as e:
                    print(e)
                    pass
                if len(generated_table) >= self.num_samples:
                    break
        # Save to wandb
        use_table=False
        if use_table:
            df = pd.DataFrame(generated_table)
            markdown_string = df.to_markdown()
        else:
            examples = []
            for entry in generated_table:
                #entry = generated_table[0]
                markdown_example = "\n".join([f"#### [{key}] \n {val[:500]}" for key, val in entry.items()])
                examples.append(markdown_example)
            markdown_string = f"\n \n --- \n".join(examples)

        #df['article_text_all'] = df['article_text_all'].str[:50]
        trainer.logger.experiment.add_text("Example Generations",markdown_string, global_step=trainer.global_step)
        #trainer.logger.experiment.log({"Example Generations": wandb.Table(dataframe=df)})