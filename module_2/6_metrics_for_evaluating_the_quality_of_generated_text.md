# Study Notes: Metrics for Evaluating the Quality of Generated Text

## Objectives of the Lesson

After completing the lesson, you will be able to:

- Describe the concept of **perplexity**.
- Evaluate the quality of text using **precision** and **recall** in text generation.
- Implement **evaluation metrics** for generated text.

## Overview

- **Generative AI and Large Language Models (LLMs)** are widely used to generate text, images, and more.
- Their success is measured by their ability to generate **consistent** and **contextually relevant text**.
- **Key Challenge**: How to measure and evaluate the performance of these models accurately.

## Perplexity

### Definition

- A metric used to evaluate the efficiency of **LLMs** and **Generative AI models**.
- Reflects how **uncertain** or **surprised** a model is when predicting the next word in a sequence.

### Calculation

1. Given a text corpus, tools assign **probabilities** to word sequences to gauge the likelihood of specific sequences occurring.
2. Perplexity is calculated as:
   - The **exponent** of the loss obtained from the model.
   - Derived from the **average cross-entropy loss** for all tokens in a sequence.

### Steps

1. Parse a sentence and compute the **conditional probability** for each word based on preceding words.
2. These probabilities are combined into an **estimated likelihood (Q)**.
3. Apply the **exponential function** to the **average cross-entropy loss** to compute **perplexity**.

### Interpretation

- **Lower Perplexity**: Indicates better performance and higher alignment between predicted and actual text distributions.
- **Higher Perplexity**: Indicates poorer performance and greater discrepancy.

## Cross-Entropy Loss

- Used to measure the **difference** between the predicted distribution and the true distribution.
- **As the predicted distribution aligns with the true distribution**, the cross-entropy loss decreases.
- When the predicted and actual distributions coincide perfectly, **loss reaches zero**.

## Metrics for Evaluating Text Generation

### Perplexity

- Provides an **overall measure of model performance** but does not capture text quality nuances.
- Typically used to evaluate how well a model has learned the **training set**.

## Precision and Recall in Machine Translation

### Precision

- Measures the **accuracy** of generated text.
- Formula:
  \[
  \text{Precision} = \frac{\text{CountMatch}}{\text{CountGenerated}}
  \]
  Where:
  - **CountMatch**: Number of matching n-grams between generated and reference text.
  - **CountGenerated**: Total number of n-grams in the generated text.

### Recall

- Measures the **completeness** of generated text.
- Formula:
  \[
  \text{Recall} = \frac{\text{CountMatch}}{\text{CountReference}}
  \]
  Where:
  - **CountReference**: Total number of n-grams in the reference text.

### F1 Score

- The **harmonic mean** of precision and recall.
- Used to judge model performance based on both accuracy and completeness.

## Example of N-Gram Matching

- **Unigram Matching**:
  - Words like "The," "Cat," "On" matched between generated and reference text.
  - Unmatched words do not contribute.
- **Bigram Matching**:
  - Example: Matching "on the" increases bigram count.

### Insights

- Precision, recall, and n-gram matching help evaluate the **quality** of text and are especially useful in tasks with multiple valid outputs, like **machine translation**.

## Other Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)

- Measures **similarity** between generated and reference text based on **n-gram matching**.
- Commonly used in machine translation.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- Evaluates generated text based on **recall**.

### METEOR

- Focuses on **precision, recall**, and **alignment** for evaluating translations.

## Popular Libraries for Evaluation Metrics

1. **NLTK (Natural Language Toolkit)**:
   - Provides BLEU and METEOR implementations.
2. **PyTorch**:
   - Includes metrics like **perplexity** and **cross-entropy loss**.
3. **Other Libraries**:
   - BLEU and ROUGE implementations for broader usage.

## Practical Example: Using NLTK to Calculate BLEU

### Steps:

1. Define a function to create **references** and **hypothesis lists** in the required format.
2. Use `sentence_bleu()` to calculate the BLEU score.
3. Return the BLEU score for generated translations.

## Recap

- **Perplexity** is a key metric to evaluate the **efficiency** of LLMs.
- It employs **cross-entropy loss** to measure discrepancies between distributions.
- **Precision and Recall** are used to evaluate the **accuracy** and **completeness** of generated text.
- Additional metrics (BLEU, ROUGE, METEOR) provide comprehensive ways to assess generated text quality.
- Libraries like **NLTK** and **PyTorch** provide accessible tools for implementing these metrics.
