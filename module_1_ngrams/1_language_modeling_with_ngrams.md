### Study Notes: Language Modeling with N-Grams

#### Objectives of the Lesson
- Understand and explain the concepts of **bi-gram** and **tri-gram** models.
- Learn how to describe and implement **language modeling with n-grams**.

---

#### Key Concepts of N-Grams

1. **N-Grams Definition**:
   - **N-grams** are sequences of `n` words used to model language by considering context from prior words.

2. **Examples**:
   - Phrases like **"I like"** and **"I hate"** illustrate how context influences word choice.

3. **Contextual Word Prediction**:
   - For **"I like ___"**, "vacation" is a more likely completion than "surgery."
   - For **"I hate ___"**, "surgery" is more commonly associated.

4. **Key Insight**:
   - Word choices are influenced by the **context** provided by preceding words.

---

#### Bi-Gram Model
1. **Definition**:
   - A **bi-gram** is a **conditional probability model** where the context size is **one** (i.e., considers only the **immediate previous word**).
   - Predicts the probability of the next word given the previous word.

2. **Example**:
   - Given the phrase **"I like"**, predict the next word using conditional probability:
     - Count occurrences of possible follow-ups for **"like"**.
     - E.g., If "surgery" never follows "like," its probability is **zero**, but if "vacation" always follows "like," its probability is **one**.

3. **Mechanics**:
   - **Context word** (e.g., "like") influences the prediction of the next word.

---

#### Tri-Gram Model
1. **Definition**:
   - A **tri-gram** is a **conditional probability model** with a **context size of two**, meaning it uses the **two previous words** to predict the next word.

2. **Example**:
   - For the phrase **"Surgeons like surgery"**, the tri-gram model predicts:
     - If **word 1 = Surgeons** and **word 2 = like**, then **word 3 = surgery** has a high probability.

3. **Advantages**:
   - Improves on bi-gram limitations by considering a larger context.
   - Incorporates more information from prior words to improve predictions.

---

#### Generalizing to N-Gram Models
1. **Definition**:
   - **N-gram models** extend the concept of bi-grams and tri-grams to use an **arbitrary context size** (`n-1` previous words).

2. **Context Vector**:
   - In neural networks, the **context vector** is defined as:
     - **Product of context size and vocabulary size**.
   - Example: For a vocabulary of 6 words and a context size of 2, the input vector dimension is `6 × 2 = 12`.

3. **Challenges**:
   - Calculating probabilities becomes complex with larger context sizes.

4. **Neural Networks**:
   - Neural networks address this challenge by **approximating probabilities** and using the **softmax function** to predict the next word.

---

#### Language Modeling Process with N-Grams
1. **Using N-Grams**:
   - Predict the next word (e.g., word at position `t`) based on the previous `n-1` words.
   - Example:
     - For a tri-gram, predict **word 3** using **word 1** and **word 2**.

2. **Optimization**:
   - Use the **argmax function** to select the word with the highest probability.

3. **Limitations of Feedforward Networks**:
   - Feedforward neural networks used for n-gram models:
     - Ignore word order beyond the context size.
     - Lack mechanisms to capture long-range dependencies or word positions effectively.
   - Modern neural networks (e.g., transformers) address these limitations.

---

#### Practical Example: Neural Network for N-Gram Model
1. **Setup**:
   - Vocabulary size: 6 words.
   - Context size: 2 words.
   - Input dimension: `6 × 2 = 12`.

2. **Output Layer**:
   - Predict one of six possible outputs (corresponding to six words in the vocabulary).
   - Output layer contains six neurons.

3. **Embedding Vectors**:
   - Use concatenation of embedding vectors for the context representation instead of raw one-hot encoding.

---

#### Key Takeaways from the Lesson
1. **Bi-Gram Model**:
   - Uses one previous word for prediction.
   - Conditional probability model with context size of 1.

2. **Tri-Gram Model**:
   - Uses two previous words for prediction.
   - Improves predictions by considering a larger context.

3. **N-Gram Model**:
   - Generalization of bi-grams and tri-grams to arbitrary context sizes.

4. **Neural Networks for N-Grams**:
   - Approximate probabilities for larger context sizes using embedding vectors.
   - Use **softmax** for probability estimation.
   - Context vector dimension = `context size × vocabulary size`.

5. **Modern Neural Networks**:
   - Address limitations of n-gram models by capturing long-range dependencies and word order.

