from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset

# Model and dataset configuration
model_name = "dunzhang/stella_en_400M_v5"
# Load the model
model = SentenceTransformer(model_name, device='cuda', trust_remote_code=True).to('cuda')
print("Model loaded successfully!")

# Load the dataset
fineweb_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
print("Dataset loaded successfully!")

# Convert dataset to sentence-transformers format
train_examples = [InputExample(texts=[data['text'], data['text']], label=1.0) for data in fineweb_dataset.take(1000)]

print("Start Data Training...")
# Define DataLoader and Loss Function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
train_loss = losses.CosineSimilarityLoss(model=model)

# Set number of epochs and warmup steps
num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps

print("Fine-tuning model...")
# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps
)

# Save the fine-tuned model
model.save("output/stella_finetuned_model")