### Study Notes: Word2Vec - Skip-Gram Model and Pretrained Embeddings

### Objectives of the Lesson

- Understand the **Skip-Gram model** in Word2Vec.
- Learn how to create and train a Skip-Gram model in PyTorch.
- Explore **pretrained embeddings** like **GloVe** and their integration into NLP tasks.

### Key Concepts

#### 1. Skip-Gram Model

- **Definition**: The Skip-Gram model is the reverse of the **Continuous Bag of Words (CBOW)** model. It predicts the **surrounding context words** given a specific **target word**.
- **Purpose**: Generates word embeddings by predicting context words based on a target word.

#### 2. How Skip-Gram Works

- **Prediction Mechanism**:
  - Predicts words at positions **t-1** and **t+1** for a given target word at position **t**.
  - Example Sentence: _"She exercises every day"_.
    - **At t=1**:
      - Target: **"exercises"**.
      - Predict: **"she"**, **"every"**.
    - **At t=2**:
      - Target: **"every"**.
      - Predict: **"exercises"**, **"day"**.
- **One-Hot Encoding**:
  - Target words are encoded as **one-hot vectors**.
  - Example: The word **"exercises"** is represented as a vector with a value of 1 for **"exercises"** and 0 for all other words.

#### 3. Comparison with CBOW

- **CBOW**: Predicts a target word based on surrounding context words.
- **Skip-Gram**: Predicts surrounding context words for a given target word.

#### 4. Training the Skip-Gram Model

- **Goal**: Train the model such that the output probabilities for context words are maximized.
- **Simplification**:
  - Instead of predicting all context words together, Skip-Gram predicts one context word at a time.
  - This breaks the complex task into smaller, manageable parts.

### Neural Network for Skip-Gram

#### 1. Architecture

- **Input**:
  - One-hot encoded vector for the target word.
- **Embedding Layer**:
  - Converts the input vector into a dense embedding vector.
- **Output**:
  - Predicts the probability distribution over the vocabulary for the context words.

#### 2. Implementation in PyTorch

- **Initialization**:
  - Define an **embedding layer** using `nn.Embedding`:
    - Input: Vocabulary size.
    - Output: Embedding dimension.
  - Define a **fully connected (FC) layer**:
    - Input: Embedding dimension.
    - Output: Vocabulary size.
- **Forward Pass**:
  - Input target word → Embedding Layer → Activation Function → FC Layer.
- **Prediction**:
  - Output probabilities indicate the most likely context words for the given target.

### Building and Training the Skip-Gram Model

#### 1. Dataset Preparation

- **Context-Target Pairs**:
  - Use a sliding window to create context-target pairs.
  - Example:
    - Sentence: _"She exercises every morning"_.
    - Pairs:
      - Target: **"exercises"**, Context: **"she"**, **"every"**.
      - Target: **"every"**, Context: **"exercises"**, **"morning"**.
- **Data Segmentation**:
  - Break down the full context into smaller parts for training.

#### 2. Data Loader and Collate Function

- **Flatten Data**:
  - Combine target and context pairs into a flattened dataset.
- **Data Loader**:
  - Use PyTorch's `DataLoader` to prepare batches for training.

#### 3. Training Setup

- **Define Components**:
  - **Loss Function**: Use `CrossEntropyLoss`.
  - **Optimizer**: Use an optimizer like `Adam` to update model weights.
  - **Learning Rate Scheduler**: Adjust the learning rate during training.
- **Training Function**:
  - Train the model for a specified number of epochs.
  - Monitor average losses for each epoch.

#### 4. Training Output

- **Trained Model**: Use the trained model to generate predictions.
- **Word Embeddings**:
  - Retrieve the **weights** from the embedding layer, which represent the word embeddings.
  - To get the embedding vector for a specific word, use its index in the vocabulary.

### Pretrained Embeddings: GloVe

#### 1. Introduction to GloVe

- **Definition**: GloVe (Global Vectors) is a pretrained word embedding model that uses large-scale data to generate word vectors.
- **Integration**:
  - Available through PyTorch's `torchtext.vocab`.

#### 2. Using GloVe in PyTorch

- **Initialization**:
  - Load GloVe embeddings with `torchtext.vocab.GloVe`.
  - Example: `GloVe(name='6B')` to load a specific pretrained vector set.
- **Custom Vocabulary**:
  - Match GloVe embeddings to a custom vocabulary using token indices.

#### 3. Applications in NLP

- **Text Classification**:
  - Integrate GloVe embeddings into a PyTorch model for classification tasks.
- **Fine-Tuning**:
  - Option to freeze or fine-tune GloVe embeddings during training, depending on dataset size.

### Key Takeaways

1. **Skip-Gram Model**:

   - Predicts context words given a target word.
   - Operates in contrast to the CBOW model, which predicts a target word from context words.

2. **Training Process**:

   - Define context-target pairs, prepare data loaders, and train the model using PyTorch.
   - Retrieve word embeddings as the model's weights after training.

3. **Pretrained Embeddings**:

   - GloVe provides high-quality word embeddings trained on large-scale data.
   - Useful for various NLP tasks like text classification.

4. **Model Simplification**:
   - Skip-Gram simplifies context prediction by breaking it into smaller tasks, improving training efficiency and effectiveness.
