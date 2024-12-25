### Study Notes: N-Grams as Neural Networks with PyTorch

#### Objectives of the Lesson
- Learn how to create an **n-gram model** in PyTorch and train it.
- Use n-grams as neural networks to predict words within a given context.

---

### Key Concepts and Steps

#### 1. Embedding Layer
- **Embedding Layer in PyTorch**:
  - Created with an **arbitrary vocabulary size**.
  - Parameters to set:
    - **Embedding dimension**: Defines the size of the embedding vectors.
    - **Context size**: Specifies how many previous words (n-1) are considered in the model.

#### 2. Context Vector Formation
- **Input Dimension**:
  - The next layer's input dimension is the product of the **context vector embedding dimension** and the **context size**.
  - Example:
    - Context size = 2, Embedding dimension = 3.
    - Output: 2 embedding vectors of dimension 3.
  
- **Reshaping**:
  - Use the **reshape** method to concatenate the embedding vectors.
  - The reshaped **context vector** is then used as input for the next layer.

#### 3. N-Gram Language Model as a Classification Task
- **Model Architecture**:
  - The n-gram model is essentially a **classification model**.
  - Components:
    - **Context vector**: Provides input to the model.
    - **Extra hidden layer**: Enhances performance by introducing non-linearity.

- **Sliding Window Mechanism**:
  - The n-gram model predicts words surrounding a target word by sliding a context window across the sequence.
  - In a **bi-gram model**:
    - Predictions for a word at position `t` are based on words at positions `t-1` and `t-2`.
    - Start prediction at `t = 3` to avoid negative indices.

---

### Example: Prediction Using N-Gram Models
- **Phrase**: *"I like vacations"*.
  - **Context**:
    - At `t=3`, the context is **"I like"** (blue).
    - Predicted target word: **"vacations"** (red).
  - **Sliding Window**:
    - At `t=4`, context updates to **"like vacations"**.
    - The prediction process repeats for the entire sequence.

- **Windowing for Batch Creation**:
  - Use windowing to generate **batches** of context and target words.
  - Implemented using a **for loop** to iterate through the context window, sliding it forward to capture successive targets and contexts.

---

### Implementing the N-Gram Model in PyTorch

#### 1. Dataset Preparation
- **Toy Dataset**:
  - Create a small dataset as a list instead of using a dataset object.
  - Use a pipeline to convert text into **token indices**.

- **Padding**:
  - Use padding tokens to ensure consistent input shapes.
  - Example: Pad with previous values to maintain alignment.

#### 2. Training the Model
- **Key Performance Indicator (KPI)**:
  - Focus on **loss** rather than **accuracy** as the primary metric for training.

- **Training Steps**:
  1. Convert token indices into PyTorch tensors.
  2. Make predictions with the model.
  3. Select the index with the highest value (argmax).
  4. Map the predicted index back to its corresponding word.

#### 3. Decoding Predictions
- **Index-to-Token Mapping**:
  - Use `vocab.get_itos` to create a list where:
    - Each element corresponds to a word.
    - The index in the list corresponds to the token index.
  - This mapping translates model outputs into **human-readable words**.

---

### Key Highlights

1. **N-Gram Model**:
   - Allows for **arbitrary context size**.
   - Functions as a **classification model** using context vectors.

2. **Sliding Window Mechanism**:
   - Predicts words by incrementally shifting a window over the sequence.

3. **PyTorch Implementation**:
   - Focus on reshaping embeddings and creating context vectors for input.
   - Train the model by minimizing **loss**, not accuracy.

4. **Mapping Predictions to Words**:
   - Use the **index-to-token mapping** to convert numerical outputs to words.

5. **Applications**:
   - Generate sequences of words by predicting one word at a time within a context window.