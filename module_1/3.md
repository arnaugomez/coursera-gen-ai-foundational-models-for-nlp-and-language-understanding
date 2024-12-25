### Study Notes: Neural Network Training and Cross-Entropy Loss with TorchText

---

#### **Introduction**
- **Topic:** Document Categorization Training with TorchText
- **Goals of the Lesson:**
  - Understand **cross-entropy loss**.
  - Learn how **optimization reduces losses** in a neural network model.

---

#### **Neural Networks and Learnable Parameters**
- **Neural Networks:**
  - Operate via **matrix and vector operations** known as **learnable parameters**.
  - Networks can have **millions to trillions** of parameters, collectively represented as **Theta (θ)**.
  
- **Training Process:**
  - Learnable parameters are **fine-tuned** to enhance performance.
  - Guided by the **loss function**, which measures the model's accuracy.

---

#### **Loss Function**
- **Purpose:**
  - Measures the **discrepancy** between:
    - Predicted output: \( \hat{y} \)
    - Actual label: \( y \)
  - Goal: Minimize this discrepancy.

- **Cross-Entropy Loss:**
  - Calculated as a function of **Theta (θ)**.
  - Encourages alignment between the **true class labels (y)** and **predicted labels (\( \hat{y} \))**.

---

#### **Prediction Process**
- Input: **Tokenized text** processed through the pipeline.
- Neural network outputs a **vector of logits**:
  - Each logit represents the likelihood of an input belonging to a specific category.
- Categories example: "World," "Sports," "Business," "Science & Technology."

---

#### **Softmax Transformation**
- Converts **logits** into **probabilities**:
  1. **Exponentiate** each logit to ensure positivity.
  2. Normalize against the sum of all logits.
- Produces a **conditional probability distribution** for each class.

---

#### **Calculating Cross-Entropy Loss**
- **Steps:**
  1. Logarithms are applied to both distributions.
  2. Compute the **KL divergence** (difference between the two distributions).
  3. Focus on the **cross-entropy term**, which depends on Theta (θ).
  
- **Monte Carlo Sampling:**
  - A technique to approximate cross-entropy loss.
  - Average the loss over a set of **sample predictions and true labels**.

---

#### **Optimization**
- **Purpose:** Reduce loss and improve model accuracy.

- **Gradient Descent:**
  - Updates parameters \(\theta_{k+1}\) using:
    \[
    \theta_{k+1} = \theta_k - \eta \cdot \nabla L
    \]
    - \( \eta \): Learning rate.
    - \( \nabla L \): Gradient of the loss function.
  - Process:
    1. Start with an initial guess for \(\theta\).
    2. Iteratively refine \(\theta\) by minimizing the loss.
    3. Loss decreases incrementally, improving accuracy.

- **Visualization:**
  - Loss surface: Depicts loss across parameter values.
  - Optimization involves **navigating this surface** to find the **minimum loss point**.

---

#### **Implementation in PyTorch**
1. **Model Setup:**
   - Define a text classification model.
   - Use PyTorch's **cross-entropy loss function**.
   
2. **Optimizer:**
   - Initialize **Stochastic Gradient Descent (SGD)** with:
     - Learning Rate (\( LR \)): 0.1
     - Optional: Use a **scheduler** to reduce LR over epochs.

3. **Training Steps:**
   - Reset gradients to zero.
   - Make predictions using the model.
   - Compute the loss using the **loss function**.
   - Use **backpropagation** to calculate gradients.
   - Apply **gradient clipping** to stabilize training.
   - Update model parameters.

---

#### **Data Partitioning**
- Typical data split:
  - **Training Set:** For learning.
  - **Validation Set:** For hyperparameter tuning.
  - **Test Set:** For evaluating real-world performance.

---

#### **Summary**
1. **Neural Networks:**
   - Operate through learnable parameters adjusted via loss minimization.
2. **Cross-Entropy Loss:**
   - Guides parameter adjustments to align predictions with actual labels.
3. **Optimization:**
   - Gradient descent and PyTorch methods effectively reduce loss.
4. **Monte Carlo Sampling:**
   - Approximates loss when distributions are unknown.
5. **Practical Application:**
   - Train a model with PyTorch using tokenized text input to predict categories.

