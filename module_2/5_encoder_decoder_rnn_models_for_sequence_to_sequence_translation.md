# Study Notes: Encoder-Decoder RNN Models for Sequence to Sequence Translation

## Overview

- This lesson introduces the **Encoder-Decoder RNN Architecture** for sequence-to-sequence models, particularly in translation tasks.
- After completing the video, you will be able to implement an encoder-decoder RNN model in PyTorch.

## **Key Concepts**

### **Sequence-to-Sequence Models with RNNs**

- **Input and Output**: RNN-based sequence-to-sequence models take an input sequence `x` and generate an output sequence `y`.
- **Variable Lengths**: The input sequence (`x`) and output sequence (`y`) do not need to be of the same length.

### **Encoder-Decoder Architecture**

- **Purpose**: Introduced to handle sequence-to-sequence problems, such as translation.
- **Components**:
  - **Encoder**:
    - Encodes the input sequence into a context vector (hidden state).
    - A series of RNNs process input tokens sequentially, passing hidden states forward.
  - **Decoder**:
    - Uses the context vector from the encoder to generate the output sequence, one token at a time.
    - Works autoregressively: each generated token is fed as input to the next RNN step.

## **Encoder**

### **Structure of the Encoder**

1. **Input Processing**:
   - Each token in the input sequence is transformed into an embedded vector using an **embedding layer**.
2. **RNN Operations**:
   - Embedded vectors pass through an RNN cell (e.g., LSTM) to produce hidden states.
   - Hidden states are sequentially passed to the next RNN cell in the encoder.
3. **Final Output**:
   - The last hidden state (context vector) is passed to the decoder.

### **Building the Encoder in PyTorch**

- Create a class inheriting from `torch.nn.Module`.
- Key layers:
  - **Embedding Layer**: Maps input tokens to embedded vectors.
  - **LSTM Layer**: Processes the embedded vectors into hidden and cell states.
  - **Dropout Layer**: Used to improve model performance by preventing overfitting.
- **Forward Method**:
  - Input: Processes tokens through the embedding layer.
  - Output: Returns hidden and cell states for the decoder.

## **Decoder**

### **Structure of the Decoder**

1. **Input Processing**:
   - Starts with the last hidden and cell states from the encoder.
   - First token is usually a special "start" token.
2. **Autoregressive Generation**:
   - For each step, the decoder generates one token using:
     - Previous token (as input).
     - Hidden and cell states from the prior step.
   - Outputs are passed through a linear layer to produce token probabilities.
3. **Final Output**:
   - Sequence of predicted tokens is generated until a special "end" token is produced.

### **Building the Decoder in PyTorch**

- Create a class inheriting from `torch.nn.Module`.
- Key components:
  - **Embedding Layer**: Maps output tokens to dense vectors.
  - **LSTM Layer**: Processes embedded vectors to produce hidden states.
  - **Linear Layer**: Maps hidden states to token probabilities.
  - **Softmax Activation**: Converts outputs to probabilities for each token.
- **Parameters**:
  - **Output Dim**: Target vocabulary size.
  - **Embedding Dim (m_dim)**: Dimensionality of embedding vectors.
  - **Hidden Dim (hid_dim)**: Dimensionality of hidden states.
  - **Number of Layers (n_layers)**: Number of LSTM layers.
  - **Dropout**: Probability for dropout regularization.

## **Sequence-to-Sequence Model**

### **Architecture**

- Combines the encoder and decoder into a unified framework.
- Inputs:
  - Source sequence (`src`).
  - Target sequence (`trg`).
  - Teacher forcing ratio.
- **Teacher Forcing**:
  - A training technique where the ground truth token is used as input to the decoder instead of the predicted token.
  - Boosts model training by providing the correct context at each step.

### **Implementation in PyTorch**

1. **Define Sequence-to-Sequence Class**:
   - Inherits from `torch.nn.Module`.
   - Initializes the encoder, decoder, and other parameters.
2. **Forward Method**:
   - Passes the source sequence through the encoder.
   - Initializes the decoder’s hidden and cell states using the encoder’s output.
   - Generates predictions for each step in the target sequence:
     - Determines whether to use teacher forcing.
     - Collects output probabilities at each step.
3. **Output**:
   - Tensor containing predictions for each time step.

## **Key Layers and Parameters**

| Component           | Purpose                                    | Parameters                                                                    |
| ------------------- | ------------------------------------------ | ----------------------------------------------------------------------------- |
| **Embedding Layer** | Converts tokens into dense vectors.        | **Input**: Vocabulary size; **Output**: Embedding dimension (`m_dim`).        |
| **LSTM Layer**      | Processes sequences to produce states.     | **Input**: Embedding vectors; **Output**: Hidden (`hid_dim`) and cell states. |
| **Linear Layer**    | Maps hidden states to token probabilities. | **Input**: Hidden states; **Output**: Token probabilities.                    |
| **Dropout Layer**   | Prevents overfitting.                      | Dropout probability.                                                          |
| **Softmax**         | Converts logits to probabilities.          | Produces a probability distribution over the target vocabulary.               |

## **Training with Teacher Forcing**

### **Definition**

- During training, the decoder uses the correct target token as input instead of the predicted token.
- This ensures faster convergence by providing the model with accurate context.

### **Process**

1. Initialize batch size, target length, and target vocabulary size.
2. Extract hidden and cell states from the encoder.
3. Start decoding with the first target token.
4. For each time step:
   - Pass input token and prior states to the decoder.
   - Decide whether to apply teacher forcing based on the teacher forcing ratio.

## **Recap of Key Points**

1. **RNNs for Sequence-to-Sequence Models**:
   - Receive an input sequence and generate an output sequence.
2. **Encoder-Decoder Architecture**:
   - Encodes input into a context vector and decodes it into an output sequence.
3. **PyTorch Implementation**:
   - Use `torch.nn.Module` to define the encoder, decoder, and sequence-to-sequence model.
4. **Teacher Forcing**:
   - Speeds up training by using the ground truth as input to the decoder.

## **Applications**

- Translation (e.g., English to French).
- Text summarization.
- Speech recognition.
- Question answering systems.
