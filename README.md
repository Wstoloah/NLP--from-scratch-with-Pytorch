# NLP from Scratch with PyTorch

This project demonstrates how to build a Name Classification model using Recurrent Neural Networks (RNNs) with PyTorch. The model predicts the nationality of a given name based on its spelling. The project is designed to be a hands-on learning experience for understanding NLP concepts and PyTorch implementation.

## Features
- **Custom Dataset**: Processes name data from text files.
- **RNN Variants**: Supports LSTM, GRU, and vanilla RNN architectures.
- **Bidirectional RNNs**: Includes support for bidirectional RNNs.
- **Training and Evaluation**: Includes training, evaluation, and visualization of results.
- **Sample Predictions**: Predicts the nationality of sample names with confidence scores.

## Project Structure
```
NLP--from-scratch-with-Pytorch
├── data/
│   ├── names/
│       ├── Arabic.txt
│       ├── Chinese.txt
│       ├── ...
├── src/
│   ├── dataset.py
│   ├── main.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   ├── name_classification.ipynb
├── requirements.txt
├── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Wstoloah/Name-classification-with-RNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Name-classification-with-RNN
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset:
   - Ensure the `data/names/` directory contains `.txt` files with names categorized by nationality. 
     Download link : [here](https://download.pytorch.org/tutorial/data.zip)

2. Train the model:
   ```bash
   python src/main.py
   ```

3. View results:
   - Training history plots
   - Classification report
   - Confusion matrix

4. Test predictions:
   - The script includes sample names for testing nationality predictions.

## Example Output
```
Sample predictions:
Smith      -> English      (confidence: 0.507)
Garcia     -> Italian      (confidence: 0.777)
Wang       -> Chinese      (confidence: 0.479)
Singh      -> English      (confidence: 0.371)
Mueller    -> German       (confidence: 0.314)
Rossi      -> Italian      (confidence: 0.549)
Petrov     -> Russian      (confidence: 0.993)

Final best accuracy: 78.19%
```

## Files
- `src/dataset.py`: Handles data loading and preprocessing.
- `src/model.py`: Defines the RNN model architecture.
- `src/train.py`: Contains the training loop and evaluation logic.
- `src/utils.py`: Utility functions for reproducibility and data processing.
- `src/main.py`: Main script to train and test the model.

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Acknowledgments
The dataset and inspiration for this project are derived from the [PyTorch tutorials](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial) and other open-source resources.
