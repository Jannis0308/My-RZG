import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

MODEL_PATH = "./fine_tuned_gpt2"
NEW_DATA_PATH = "./new_data.json"

# Modell und Tokenizer laden
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

# Padding-Token hinzufügen (falls erforderlich)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Neue Daten laden
with open(NEW_DATA_PATH, "r") as f:
    new_data = json.load(f)

# Daten in ein Dataset-Format umwandeln
texts = [entry["input"] for entry in new_data]
dataset = Dataset.from_dict({"text": texts})


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Trainingseinstellungen
training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=torch.cuda.is_available()
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Training starten
trainer.train()

# Modell speichern
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print("Neues Training abgeschlossen.")
