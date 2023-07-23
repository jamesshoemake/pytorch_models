"""
train a pytorch image classification model using device-agnostic code.
"""
import os
import torch
from torchvision import transforms
import data_setup, engine, model, utils 
from timeit import default_timer as timer 

# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# get dataloaders and class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform= data_transform,
    batch_size = BATCH_SIZE
)

# Recreate an instance of TinyVGG
model = model.TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=HIDDEN_UNITS, 
                  output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Train model_0 
model_results = engine.train(model=model, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS,
                        device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model
utils.save_model(model=model,
           target_dir="models",
           model_name="05_going_modular_script_mode_tinyvgg_model.pth")
