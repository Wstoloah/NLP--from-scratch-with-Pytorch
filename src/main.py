# Name Classification using RNN with PyTorch

import os
import torch
from torch.utils.data import DataLoader


from dataset import NamesDataset, collate_fn
from model import EnhancedCharRNN, predict_nationality
from train import Trainer
from utils import set_seed, n_letters

# Set random seeds for reproducibility
set_seed(42)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if __name__ == "__main__":

    # Load data 
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # path to your data folder
    data_path = os.path.join(root, "data", "names")
    
    try:
        dataset = NamesDataset(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the data directory path is correct and contains .txt files")
        exit(1)
    
    # Split dataset
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    # Model parameters
    hidden_size = 64
    num_classes = len(dataset.labels_uniq)
    
    # Create model
    model = EnhancedCharRNN(
        input_size=n_letters,
        hidden_size=hidden_size,
        output_size=num_classes,
        num_layers=2,
        rnn_type='LSTM',  # Try 'LSTM', 'GRU', or 'RNN'
        dropout=0.2,
        bidirectional=True
    ).to(device)
    
    print(f"\nModel: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, test_loader, device, dataset.labels_uniq)
    
    # Train model
    print("\nStarting training...")
    best_accuracy = trainer.train(epochs=20, lr=0.001)
    
    # Plot results
    trainer.plot_training_history()
    trainer.print_classification_report()
    trainer.plot_confusion_matrix()
    
    # Test predictions on sample names
    test_names = ["Smith", "Garcia", "Wang", "Singh", "Rossi", "Petrov", "Fatima", "Jules"]
    print("\nSample predictions:")
    for name in test_names:
        predicted, confidence = predict_nationality(model, name, dataset.labels_uniq, device)
        print(f"{name:10} -> {predicted:12} (confidence: {confidence:.3f})")
    
    # Ensure results directory exists
    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "training_results.txt")

    # Write results to file
    with open(results_file, "w") as f:
        f.write("Training Results\n")
        f.write("================\n")
        f.write(f"Final best accuracy: {best_accuracy:.2f}%\n\n")

        f.write("Sample Predictions:\n")
        for name in test_names:
            predicted, confidence = predict_nationality(model, name, dataset.labels_uniq, device)
            result_line = f"{name:10} -> {predicted:12} (confidence: {confidence:.3f})\n"
            f.write(result_line)
            print(result_line.strip())  # Print to console as well

    print(f"\nTraining results saved to {results_file}")