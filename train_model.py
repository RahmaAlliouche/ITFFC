import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Load the data
data_path = "c:/Users/rahma/Documents/S2/ITFFC_M/model/text_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"CSV file not found at {data_path}")

data = pd.read_csv(data_path, on_bad_lines='warn')  # Skips bad lines and warns you

# Ensure 'text' and 'label' columns exist
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("The dataset must contain 'text' and 'label' columns.")

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(), data['label'].tolist(), test_size=0.2, random_state=42
)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Define Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Prepare datasets
train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure models are saved after each epoch
    save_total_limit=1,     # Keep only the latest checkpoint
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete! Model and tokenizer saved.")
