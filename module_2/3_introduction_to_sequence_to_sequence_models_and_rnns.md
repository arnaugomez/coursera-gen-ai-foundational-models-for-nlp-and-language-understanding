# Study Notes: Introduction to Sequence-to-Sequence Models and Recurrent Neural Networks (RNNs)

## Overview

This lesson covers the fundamentals of sequence-to-sequence models and Recurrent Neural Networks (RNNs). It explains how these models have revolutionized tasks in Natural Language Processing (NLP), such as machine translation, text summarization, and chatbots. Additionally, the lesson introduces basic RNN structures and their enhancements like GRUs and LSTMs.

## Key Topics

### **1. Sequence-to-Sequence Models**

- **Definition**: These models process sequences of variable lengths as inputs and outputs.
- **Applications**:

  - Machine translation (e.g., translating English to French).
  - Text summarization (condensing large texts into concise summaries).
  - Chatbots (generating conversational responses).
  - Code generation (AI generates appropriate code based on task descriptions).

- **Core Principles**:
  - **Input and Output**:
    - Inputs and outputs can have different lengths.
    - Examples:
      - _Sequence-to-label_: Multiple inputs → Single label (e.g., document classification).
      - _Label-to-sequence_: Single input → Full sequence (e.g., generative image models).
  - **Context Importance**:
    - Sequence representation, like embeddings, captures meaning and preserves contextual information.

### **2. Sequence Representation**

- **Challenge**: Bag-of-words models lose word order and context.
- **Solution**: Represent sequences using:
  - **One-hot encoding**.
  - **Embeddings**: Capture semantic meaning and maintain the distinction between similar sentences like:
    - "The man bites the dog."
    - "The dog bites the man."

### **3. Recurrent Neural Networks (RNNs)**

- **Definition**: Artificial neural networks that handle sequential or time-series data.
- **Key Features**:
  - Designed to "remember" past information to influence future outputs.
  - Operates under the concept of dependencies in sequences.

#### **RNN Operations**

1. **Input Layer (x_t)**:

   - RNN receives data for each timestep (e.g., receiving a new puzzle piece at each moment).

2. **Hidden State (h_t)**:

   - Functions as the network’s memory.
   - Combines past inputs and current input to update the state.
   - Activation function (usually tanh) is applied to retain information.

3. **Output Layer (z_t)**:

   - The output is computed based on the hidden state and the current input.
   - Used in various applications such as classification.

4. **Unrolling Over Time**:
   - Starts with an initial hidden state (vector of zeros).
   - Hidden states and outputs are updated iteratively for each timestep.

### **4. Training RNNs**

- **Challenges**:
  - RNNs tend to remember only short-term dependencies.
  - Difficult to train due to vanishing/exploding gradients.
- **Steps in Training**:
  1. Add **BOS** (beginning of sequence) and **EOS** (end of sequence) tags to mark sequence boundaries.
  2. Sort sentences by length to streamline batching.
  3. Add padding to shorter sequences to maintain uniform batch sizes.

### **5. RNN Enhancements**

#### **Gated Recurrent Units (GRUs)**

- **Structure**: Contains two gates:
  - **Update Gate (z)**: Determines how much of the previous hidden state is carried forward.
  - **Reset Gate (r)**: Decides how much past information is disregarded.
- **Purpose**: Simplifies the RNN while maintaining the ability to learn dependencies over time.

#### **Long Short-Term Memory (LSTMs)**

- **Structure**: Comprises three gates:
  - **Input Gate**: Decides which information to update.
  - **Forget Gate**: Filters out irrelevant past information.
  - **Output Gate**: Determines which information is output at a timestep.
- **Advantages**:
  - Extends memory capacity by integrating short-term and long-term dependencies.
  - Selectively retains and transports crucial data through time.

### **6. Key Concepts in RNN Models**

- **Conditional Distributions**:

  - If sequence elements depend on prior elements, the model must adjust probability distributions dynamically (e.g., non-IID scenarios).

- **Training Techniques**:

  - Use sorted batches and padding to improve efficiency.
  - Append padding symbols for shorter sequences to ensure consistent batch sizes.

- **Decoding Methods**:
  - **Greedy Decoding**: Selects the token with the highest score as the prediction.
  - **Top-k Sampling**: Produces more fluent text by considering multiple top tokens.

### **7. Summary**

- **Sequence-to-sequence models**: Used for tasks like machine translation, summarization, and chatbots.
- **RNNs**:
  - Suitable for sequential data.
  - Remember past information to inform future outputs.
- **Challenges**:
  - Limited memory (short-term focus).
  - Training difficulty.
- **Enhancements**:
  - GRUs and LSTMs improve memory handling and training efficiency.
