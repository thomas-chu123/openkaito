from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import torch

# Model and dataset configuration
# model_name = "dunzhang/stella_en_400M_v5"
model_name = "distiluse-base-multilingual-cased-v2"
model_size = 500000
batch_size = 8
num_epochs = 1
model_device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the model
model = SentenceTransformer(model_name, device=model_device, trust_remote_code=True).to(model_device)
print("Model loaded successfully!")

# Load the dataset
fineweb_dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
print("Dataset loaded successfully!")

# Convert dataset to sentence-transformers format
print(f"Creating InputExamples...{str(model_size)} items")
train_examples = [InputExample(texts=[data['text'], data['text']], label=1.0) for data in tqdm(fineweb_dataset.take(model_size), desc="Creating InputExamples")]

print(f"Start Data Training...{str(batch_size)} batch size")
# Define DataLoader and Loss Function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Set number of epochs and warmup steps
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps

print("Fine-tuning model...")
# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
)

# Evaluate the model
model.eval()

# Save the fine-tuned model
model.save("output/finetuned_model")