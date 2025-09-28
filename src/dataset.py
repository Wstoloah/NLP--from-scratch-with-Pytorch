import torch
from torch.utils.data import Dataset
import glob
import os

from utils import lineToTensor, unicodeToAscii

# Dataset class 
class NamesDataset(Dataset):
    """Enhanced Dataset class with better error handling and preprocessing"""
    
    def __init__(self, data_dir, min_length=2, max_length=50):
        self.data_dir = data_dir
        self.min_length = min_length
        self.max_length = max_length
        
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess data from text files"""
        text_files = glob.glob(os.path.join(self.data_dir, '*.txt'))
        
        if not text_files:
            raise ValueError(f"No .txt files found in {self.data_dir}")
        
        print(f"Found {len(text_files)} language files")
        
        # First pass: collect all labels
        labels_set = set()
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
        
        # Create label mapping
        self.labels_uniq = sorted(list(labels_set))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_uniq)}
        
        print(f"Languages: {self.labels_uniq}")
        
        # Second pass: load data
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            
            try:
                with open(filename, encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    
                for name in lines:
                    name = name.strip()
                    if not name:
                        continue
                        
                    # Preprocess name
                    clean_name = unicodeToAscii(name)
                    
                    # Filter by length
                    if self.min_length <= len(clean_name) <= self.max_length:
                        self.data.append(clean_name)
                        self.labels.append(self.label_to_idx[label])
                        
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
        print(f"Loaded {len(self.data)} names")
        
        # Print class distribution
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print the distribution of classes"""
        from collections import Counter
        label_counts = Counter(self.labels)
        print("\nClass distribution:")
        for label_idx, count in sorted(label_counts.items()):
            label_name = self.labels_uniq[label_idx]
            print(f"  {label_name}: {count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        tensor = lineToTensor(name)
        return tensor, label, name

def collate_fn(batch):
    """Collate function to handle variable-length sequences"""
    tensors, labels, names = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([t.size(0) for t in tensors])
    
    # Pad sequences
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded, labels, lengths, names
