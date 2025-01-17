import evaluate
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer



# REF: https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360
# codes: https://github.com/anyuanay/medium/blob/main/src/working_huggingface/Working_with_HuggingFace_ch3_Fine_Tuning_T5_Small_Text_Summarization_Model.ipynb?source=post_page-----3d48eb3c4360--------------------------------


# load dataset
"""
pip install -q transformers[torch] datasets
Each instance in the datasets is a dictionary with 3 keys:
text: the text for summarization.
summary: a given summary of the text.
title: the title of the text
"""
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("t5-small")
rouge = evaluate.load("rouge")

#example =  billsum['train'][0]
#for key in example:
#    print("A key of the example: \"{}\"".format(key))
#    print("The value corresponding to the key-\"{}\"\n \"{}\"".format(key, example[key]))

#pref_text = "summarize: " + example['text']
#tokenized_text = tokenizer(pref_text)
#tokenized_summary = tokenizer(example['summary'])
#tokenized_billsum = billsum.map(preprocess_function, batched=True)


def preprocess_function(examples):
    # Prepends the string "summarize: " to each document in the 'text' field of the input examples.
    # This is done to instruct the T5 model on the task it needs to perform, which in this case is summarization.
    inputs = ["summarize: " + doc for doc in examples["text"]]

    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Tokenizes the 'summary' field of the input examples to prepare the target labels for the summarization task.
    # Sets a maximum token length of 128, and truncates any text longer than this limit.
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # Assigns the tokenized labels to the 'labels' field of model_inputs.
    # The 'labels' field is used during training to calculate the loss and guide model learning.
    model_inputs["labels"] = labels["input_ids"]

    # Returns the prepared inputs and labels as a single dictionary, ready for training.
    return model_inputs


# design evaluation metrics
def compute_metrics(eval_pred):
    # Unpacks the evaluation predictions tuple into predictions and labels.
    predictions, labels = eval_pred

    # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replaces any -100 values in labels with the tokenizer's pad_token_id.
    # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Computes the ROUGE metric between the decoded predictions and decoded labels.
    # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Calculates the length of each prediction by counting the non-padding tokens.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
    result["gen_len"] = np.mean(prediction_lens)

    # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
    return {k: round(v, 4) for k, v in result.items()}



# Train
tokenized_billsum = billsum.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="my_fine_tuned_t5_small_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

trainer.save_model("my_fine_tuned_t5_small_model")


# --------------------------------------------------------
# testing

text = billsum['test'][100]['text']
text = "summarize: " + text

from transformers import pipeline

summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model")
pred = summarizer(text)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_t5_small_model")
inputs = tokenizer(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("my_fine_tuned_t5_small_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

tokenizer.decode(outputs[0], skip_special_tokens=True)
