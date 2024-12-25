# Study Notes: Converting Words to Features

## **Overview of the Lesson**

This lesson explores techniques to convert text data into numerical features suitable for machine learning models, focusing on the following concepts:

- One-Hot Encoding
- Bag of Words
- Embedding and Embedding Bags
- Implementation of embeddings in PyTorch

## **1. Introduction**

- Text data must be transformed into numerical features for machine learning models to process it.
- Common use case: **Natural Language Processing (NLP)** tasks, e.g., email classification.
  - Emails can be categorized based on:
    - Presence of specific words
    - Frequency of words
    - Contextual meaning of words.

## **2. One-Hot Encoding**

### **Definition:**

- A technique that converts **categorical data** into feature vectors that a neural network can process.

### **Process:**

1. **Representation**:
   - Create a table with:
     - **First column:** Token index
     - **Second column:** Token
     - **Third column:** One-hot encoded vector
2. **Vector Dimensions**:
   - Dimension of the vector equals the number of words in the vocabulary.
   - For a token:
     - All elements are set to zero except for the position corresponding to the token.

### **Example:**

- Vocabulary: `["I", "like", "cats"]`
- Representation:

  | Token Index | Token | One-Hot Encoded Vector |
  | ----------- | ----- | ---------------------- |
  | 0           | I     | [1, 0, 0]              |
  | 1           | like  | [0, 1, 0]              |
  | 2           | cats  | [0, 0, 1]              |

- Each token's one-hot vector becomes the feature (e.g., the vector for "I" is `[1, 0, 0]`).

## **3. Bag of Words (BoW)**

### **Definition:**

- Represents a document as the aggregate (or average) of one-hot encoded vectors.

### **Process:**

1. Combine the one-hot vectors for all tokens in a document.
2. Example:
   - Sentence: `"I like cats"`
   - Aggregate One-Hot Vector: Sum or average of individual token vectors.
   - Resulting vector represents the document.

### **Usage:**

- Used to depict the entire document or sequence.

## **4. Embeddings and Embedding Bags**

### **Embeddings**

- **Definition**: A dense representation of words in lower dimensions compared to one-hot encoding.
- **Embedding Layer**:
  - Takes a **token index** as input instead of a one-hot encoded vector.
  - Outputs an **embedding vector**.
  - Embedding weights form an **embedding matrix**, where:
    - Each row corresponds to a word.
    - Columns represent the embedding dimensions.
  - Reduces computational complexity by lowering vector dimensionality.

### **Embedding Bag Layer**

- **Definition**: A specialized layer that directly computes the sum (or average) of word embeddings for a document.
- Input:
  - Token indexes for words in the document.
- Output:
  - The sum or average of embedding vectors for the words.

## **5. PyTorch Implementation**

### **Steps to Use Embeddings in PyTorch:**

1. **Tokenize the Text:**

   - Use a tokenizer to split the dataset into tokens and map them to indexes.

2. **Create an Embedding Layer:**

   - Use the `nn.Embedding` constructor.
   - Specify:
     - Vocabulary size
     - Embedding dimension.

3. **Retrieve Embeddings:**

   - Input the token indexes of a phrase (e.g., "I like cats") into the embedding layer.
   - The output is a PyTorch tensor where:
     - Each row corresponds to the embedding vector of a token.

4. **Use Embedding Bag Layer:**

   - Use `nn.EmbeddingBag` for efficiency.
   - Inputs:
     - Token indexes
     - Offset parameter (indicates the starting position of each document in a batch).
   - Output:
     - A single embedding vector representing the entire document (sum or average of embeddings).

5. **Calculate Offsets:**
   - Use cumulative sums of token counts in each document to determine offsets for embedding bags.

## **6. Recap of Key Concepts**

### **One-Hot Encoding:**

- Converts categorical data into sparse vectors with all elements zero except one.

### **Bag of Words:**

- Represents a document as the sum/average of one-hot encoded vectors.

### **Embeddings:**

- Dense vector representation of words in lower dimensions, reducing computational requirements.

### **Embedding Bags:**

- Efficient computation of document-level embeddings by summing/averaging word embeddings.

## **Summary of PyTorch Functions**

| **PyTorch Module** | **Purpose**                                                |
| ------------------ | ---------------------------------------------------------- |
| `nn.Embedding`     | Creates embedding layers for word-level embeddings.        |
| `nn.EmbeddingBag`  | Computes document embeddings by summing/averaging vectors. |
| `input_ids()`      | Tokenizes text and maps tokens to indexes.                 |
| `cumsum()`         | Computes cumulative sums for offset calculations.          |

## **Practical Example in PyTorch:**

1. Tokenize: Convert `"I like cats"` into token indexes `[0, 1, 2]`.
2. Initialize:
   - Embedding layer: `nn.Embedding(vocab_size, embedding_dim)`
   - Embedding bag layer: `nn.EmbeddingBag(vocab_size, embedding_dim)`.
3. Retrieve Embedding:

   - Input: `[0, 1, 2]` (indexes for "I like cats").
   - Output: Tensor with embedding vectors for "I", "like", and "cats".

4. Aggregate:
   - Use Embedding Bag to compute the sum or average of the embeddings for the document.

## **Conclusion**

- Text data preprocessing is a critical step in NLP tasks.
- Techniques like one-hot encoding, bag of words, embeddings, and embedding bags offer methods to convert text into numerical formats for machine learning.
- PyTorch provides efficient tools (`nn.Embedding` and `nn.EmbeddingBag`) to implement these methods seamlessly.
