from huggingface_hub import notebook_login
notebook_login()
#
from datasets import load_dataset
from fontTools.unicodedata import block
from ply.yacc import token
from statsmodels.graphics.tukeyplot import results
# from tensorflow import optimizers
from tensorflow.compiler.tf2xla.python.xla import concatenate
from transformers import AutoTokenizer

eli5 = load_dataset("eli5_category", split="train[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)

# print(eli5["train"][0])
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

eli5 = eli5.flatten()
def preprocess_function(examples):
    out = tokenizer([" ".join(x) for x in examples["answers.text"]])
    return out

print(preprocess_function(eli5["train"][0]))

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names
)


block_size = 128


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = len(concatenated_examples[list(examples.keys())[0]])

    if total_len >= block_size:
        total_len = (total_len // block_size) * block_size

    result = {
        k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)


from transformers import DataCollatorForLanguageModeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")


from transformers import AdamWeightDecay, TFAutoModelForMaskedLM
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
tf_train_set = model.prepare_tf_dataset(
    lm_dataset["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    lm_dataset["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

import tensorflow as tf
model.compile(optimizer=optimizer)

from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(
    save_strategy='epoch',
    save_steps=200,
    output_dir="test_eli5",
    tokenizer=tokenizer,
)

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
