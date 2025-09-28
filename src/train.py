import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class Trainer:
    """Training and evaluation class"""
    
    def __init__(self, model, train_loader, test_loader, device, label_names):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.label_names = label_names
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets, lengths, _) in enumerate(pbar):
            data, targets, lengths = data.to(self.device), targets.to(self.device), lengths.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def evaluate(self, criterion):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets, lengths, _ in tqdm(self.test_loader, desc="Testing"):
                data, targets, lengths = data.to(self.device), targets.to(self.device), lengths.to(self.device)
                
                outputs = self.model(data, lengths)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, epochs, lr=0.001, weight_decay=1e-4):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_accuracy = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Evaluate
            test_loss, test_acc, predictions, targets = self.evaluate(criterion)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best accuracy: {best_accuracy:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with accuracy: {best_accuracy:.2f}%")
        
        return best_accuracy
    
    def plot_training_history(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.test_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.test_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_classification_report(self):
        """Print detailed classification report"""
        _, _, predictions, targets = self.evaluate(nn.CrossEntropyLoss())
        
        print("\nClassification Report:")
        print(classification_report(targets, predictions, 
                                  target_names=self.label_names, 
                                  digits=4))
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        _, _, predictions, targets = self.evaluate(nn.CrossEntropyLoss())
        
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
