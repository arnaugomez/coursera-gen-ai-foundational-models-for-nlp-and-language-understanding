### Study Notes: Word2Vec Introduction and CBOW Models

---

### Objectives of the Lesson
- Understand the concept of **Word2Vec**.
- Learn about **Word2Vec's Continuous Bag of Words (CBOW) model**.
- Build and train a CBOW model to predict target words using context.

---

### Key Concepts

#### 1. Word2Vec
- **Definition**: Word2Vec is short for **Word to Vector**. It is a collection of models that produce **word embeddings** or vectors.
- **Purpose**: Converts words into numerical representations that capture semantic meanings and relationships between words.
- **Example**:
  - The word "king" is closer to "man", and "queen" is closer to "woman" in the embedding space.
  - Subtracting "man" from "king" results in a vector similar to "queen", demonstrating the ability to capture relationships.

#### 2. Word Embeddings
- **What They Capture**:
  - Similarity in meaning between words.
  - Relationships between words based on context.
- **Applications**:
  - Used in **Natural Language Processing (NLP)** tasks to improve performance.
  - Replace randomly generated embeddings with meaningful vectors learned from data.

#### 3. Neural Network Structure for Word2Vec
- Components:
  - **Input layer**: Receives words or tokens as input.
  - **Embedding layer**: Learns the numerical representations of words.
  - **Output layer**: A **softmax layer** predicts context words based on the target word.
- **Training**:
  - Adjust weights in the hidden layer and output layer (denoted as **W** and **W'**) to refine the word vectors.
- **Dimensions**:
  - **Input layer neurons**: Correspond to the vocabulary size.
  - **Embedding layer size**: Determines the dimensions of the word vectors (user-defined).

#### 4. Word2Vec Predictions
- Example:
  - After training, if the model predicts **"queen"** following **"woman"** with higher probability than **"man"**, the embedding for **"queen"** will be closer to **"woman"**.
  - Similarly, the embedding for **"king"** will be closer to **"man"** than **"woman**.

---

### Continuous Bag of Words (CBOW) Model

#### 1. Overview
- **Purpose**: Predicts a **target word** based on **context words** and generates word embeddings.
- **Mechanism**:
  - Uses a **sliding window** to define context and target pairs.
  - Context words are combined into a **bag of words vector**.
  - The vector passes through the network to predict the target word.

#### 2. Example: Predicting Words Using CBOW
- Sentence: *"She exercises every morning"*.
  - Window width = 1.
  - **At t=1**:
    - Context: **"she"**, **"every"**.
    - Target: **"exercises"**.
  - **At t=2**:
    - Context: **"exercises"**, **"morning"**.
    - Target: **"every"**.

#### 3. Model Components
- **Input Dimension**:
  - Matches the number of unique words in the vocabulary.
- **Output Dimension**:
  - Matches the vocabulary size (predicts likelihood for each word in the corpus).
- **Hidden Layer**:
  - Contains word embeddings learned during training.
- **Output**:
  - Predicts the word with the highest probability for a given context.

#### 4. Training the CBOW Model
- **Goal**: Fine-tune the weights to effectively predict the target word.
- **Steps**:
  1. Encode context words as **one-hot vectors**.
  2. Feed the context vectors to the model.
  3. Adjust weights to maximize the output probability of the target word.

---

### Building the CBOW Model in PyTorch

#### 1. Model Initialization
- Define the CBOW model with:
  - **Embedding layer**: Use `nn.EmbeddingBag` to calculate the average of context word embeddings.
  - **Fully connected layer**: Use `self.fc` with:
    - Input size = embedding dimension.
    - Output size = vocabulary size.

#### 2. Forward Pass
- Steps:
  1. Input text and offsets are passed through the **embedding layer** to retrieve context embeddings.
  2. Calculate the **average embedding** of context words.
  3. Apply the **ReLU activation function** to introduce non-linearity.
  4. Pass the result through the **fully connected layer** to get predictions.

#### 3. Training Workflow
- Steps:
  1. Initialize the model.
  2. **Tokenizer**:
     - Tokenize the text.
     - Create a vocabulary from tokenized data.
  3. Set the **context size** (e.g., 2).
  4. Generate context-target pairs by sliding over the text.
  5. Create a **data pipeline** and **data loader** to handle training batches.
  6. Train the model with batches of size 64.

---

### Key Takeaways

1. **Word2Vec**:
   - A powerful technique for generating **word embeddings**.
   - Encodes semantic meanings and relationships between words in a vector space.

2. **Neural Network for Word2Vec**:
   - Input and output layers correspond to the vocabulary size.
   - Embedding layer learns word representations.

3. **CBOW Model**:
   - Predicts target words using context words.
   - Combines context words into a bag of words vector.
   - Trains the network to generate accurate word embeddings.

4. **Implementation in PyTorch**:
   - Use `nn.EmbeddingBag` for embedding calculations.
   - Train the model with context-target pairs and fine-tune weights for accurate predictions.