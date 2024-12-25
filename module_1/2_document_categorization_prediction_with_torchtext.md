# Study Notes: Document Categorization Prediction with TorchText

## **Overview of the Lesson**

This lesson focuses on creating a **document classification system** using TorchText and PyTorch. Key topics covered include:

- Understanding neural networks and their components.
- Exploring neural network hyperparameters.
- Building and training a document classifier.
- Using the AG News dataset to classify text into categories like sports, business, and technology.

## **1. Document Categorization**

- **Definition**: Automatically categorizing text documents (e.g., news articles) into predefined categories by analyzing their content.
- **Workflow**:
  - Input: Raw text.
  - Output: Predicted category (e.g., science, business, sports).

## **2. Neural Networks Overview**

### **Definition**

- A neural network is a **mathematical function** comprising:
  - **Matrix multiplications**.
  - **Activation functions**.

### **Structure**

1. **Input Layer**:

   - Example: A bag of words vector representing document text.

2. **Hidden Layers**:

   - Perform matrix multiplications and apply biases.
   - Use activation functions to compute **neurons** (transformed data at each layer).

3. **Output Layer**:

   - Outputs **logits**:
     - Numerical scores indicating the likelihood of the input belonging to each category.

4. **Argmax Function**:
   - Identifies the index of the highest logit value.
   - The index corresponds to the predicted category.

## **3. Neural Network Components**

### **Embedding Layer**:

- Converts input token indices into **dense embedding vectors**.
- These embeddings are passed to the network for further processing.

### **Learnable Parameters**:

- Weights and biases adjusted during training to minimize error.

## **4. Example Workflow for Document Categorization**

1. **Input**:

   - Text: "ESPN’s varied football coverage."
   - Tokenized into indices and converted to embeddings.

2. **Process**:

   - Embedding vectors are passed through the network.
   - The network outputs logits (scores) for each category.

3. **Output**:
   - Logits: `[3.5, 7.0, 2.1, 1.8]`.
   - Categories: `[World, Sports, Business, Science & Tech]`.
   - **Argmax Function** selects the highest value (7.0), predicting the category as **Sports**.

## **5. Neural Network Hyperparameters**

### **Definition**:

- **Externally set configurations** that define the structure and behavior of a neural network.

### **Key Hyperparameters**:

1. **Number of Layers**:
   - Input Layer → Hidden Layer(s) → Output Layer.
   - Example: Single hidden layer vs. two hidden layers.
2. **Number of Neurons**:
   - Neurons in each layer can vary.
   - Example:
     - Second hidden layer has 5 neurons in one architecture, and 4 in another.
3. **Embedding Layer Size**:
   - For the first layer, the number of neurons equals the **vocabulary size**.
4. **Output Layer Size**:
   - Always equals the **number of output classes**.

### **Selection**:

- Determined via **empirical validation** on training data.

## **6. Building a Document Classifier in PyTorch**

### **Dataset**:

- **AG News Dataset**:

  - News articles categorized into labels (e.g., sports, business, science).
  - Example:

    | **Label**          | **Text**                                     |
    | ------------------ | -------------------------------------------- |
    | Business (Label 0) | "Stock prices surged after the announcement" |
    | Science (Label 1)  | "New technology trends in AI innovation"     |

### **Pipeline Steps**:

1. **Data Preparation**:

   - Tokenize text and map tokens to indices.
   - Ensure labels are numbered starting from 0.

2. **Model Architecture**:

   - **Embedding Bag Layer**:
     - Computes a single embedding for each document by summing/averaging word embeddings.
   - **Fully Connected Layer**:
     - Converts the embedding into logits for each class.

3. **Batch Processing**:

   - Create batches of data with a batch size (e.g., 3).
   - Prepare inputs:
     - **Token indices**.
     - **Offsets** for embedding bags.

4. **Prediction Function**:
   - Takes tokenized text as input.
   - Passes text through the model to predict the category.

## **7. Model Workflow**

### **Forward Pass**

1. Input:
   - Text indices (e.g., tokenized "I like cats").
   - Offsets to specify document positions.
2. Embedding Bag:
   - Computes the sum of word embeddings for each document.
3. Fully Connected Layer:
   - Transforms embeddings into logits.
4. Argmax:
   - Identifies the class with the highest logit value.

### **Making Predictions**

- For real text:
  - Input tokenized text.
  - Process text through the model.
  - Output: Predicted category.

## **8. Summary**

### **Key Learnings**:

1. **Document Classifier**:
   - Categorizes text documents based on content using neural networks.
2. **Neural Networks**:
   - Comprised of matrix multiplications, activation functions, and learnable parameters.
3. **Hyperparameters**:
   - Define network architecture (e.g., number of layers, neurons).
4. **Argmax**:
   - Identifies the most likely category based on logits.
5. **PyTorch Implementation**:
   - Use TorchText to prepare data and build a document classifier.
   - Implement embedding bag layers for efficient text processing.

## **Practical Application in PyTorch**

1. **Data Loading**:
   - Tokenize text using TorchText and prepare labels.
2. **Model Definition**:
   - Use `nn.EmbeddingBag` for efficient embedding computation.
   - Add a fully connected layer for logits calculation.
3. **Training**:
   - Pass batches of tokenized data through the model.
   - Adjust weights and biases using a loss function.
4. **Prediction**:
   - Use the trained model to predict categories for new text.

## **Code Snippet Example**

```python
import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Load dataset
train_iter = AG_NEWS(split='train')
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[1]), train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Model definition
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding_bag(text, offsets)
        return self.fc(embedded)

# Example usage
vocab_size = len(vocab)
embed_dim = 64
num_class = 4  # Number of categories

model = TextClassificationModel(vocab_size, embed_dim, num_class)
```

### **Try It Yourself**

- Test your document classifier with new text using the trained model.
- Example:
  - Input: `"Artificial intelligence is revolutionizing industries."`
  - Output: `Science and Technology`.
