import tensorflow as tf
from datasets import load_dataset
from transformers import TFMT5ForConditionalGeneration, MT5Tokenizer, DataCollatorForSeq2Seq
from tensorflow.keras.optimizers import Adam

# REF: https://medium.com/@radicho/fine-tuning-the-multilingual-t5-model-from-huggingface-with-keras-f7f619ec5cfe

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# data processing
dataset = load_dataset("csv", data_files="train.csv")
dataset = dataset["train"].shuffle(seed=42)

def preprocess_function(examples):
    padding = "max_length"
    max_length = 200

    inputs = [ex for ex in examples["Text"]]
    targets = [ex for ex in examples["Expected"]]
    model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True)
    labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer")

# training
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    pad_to_multiple_of=64,
    return_tensors="np")

tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    batch_size=8,
    shuffle=True)

model.compile(optimizer=Adam(3e-5))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.fit(tf_train_dataset, epochs=10, callbacks=[early_stopping])