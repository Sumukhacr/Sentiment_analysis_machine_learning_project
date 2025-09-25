# Sentiment Analysis of US Airline Reviews

## Overview
This project performs sentiment analysis on a dataset of US airline reviews using a deep learning model. The goal is to classify tweets as either **positive** or **negative** sentiment. The model is built using **TensorFlow** and **Keras**, with data handling managed by **Pandas**.

## Dataset
The dataset used is `Tweets.csv`, which contains various airline-related tweets and their corresponding sentiment labels. For this analysis, we focus on two sentiments: `positive` and `negative`, filtering out `neutral` reviews.

## Key Technologies
-   **Pandas**: For data loading and preprocessing.
-   **TensorFlow/Keras**: For building and training the deep learning model.
-   **Matplotlib**: For visualizing the model's performance.

## Model Architecture
The sentiment analysis model is a **Recurrent Neural Network (RNN)**, specifically an **LSTM (Long Short-Term Memory)** network. The architecture includes:
-   An **Embedding layer** to convert words into dense vectors.
-   A **SpatialDropout1D layer** for regularization.
-   An **LSTM layer** to capture sequential information in the text.
-   A **Dropout layer** to prevent overfitting.
-   A **Dense layer** with a **sigmoid activation** function for binary classification.

## Results
The model's performance is evaluated based on **accuracy** and **loss** during training. The following plots illustrate the model's performance on both the training and validation datasets.

### Accuracy Plot
![Accuracy plot](Accuracy%20plot.jpg)
*This plot shows how the training and validation accuracy changed over the epochs.*

### Loss Plot
![Loss plot](Loss%20plot.jpg)
*This plot shows how the training and validation loss changed over the epochs.*

## How to Run the Project
1.  **Clone the repository:**
    ```
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```
2.  **Ensure you have the dataset `Tweets.csv` in the same directory.**
3.  **Install the required libraries:**
    ```
    pip install pandas tensorflow matplotlib scikit-learn
    ```
4.  **Run the Python script:**
    ```
    python your_script_name.py
    ```

## Prediction Examples
The project includes a function to predict the sentiment of new, unseen sentences.

| Sentence | Predicted Sentiment |
| :--- | :--- |
| "I enjoyed my journey on this flight." | Positive |
| "This is the worst flight experience of my life!" | Negative |

The model successfully predicts the sentiment of both a positive and a negative review, demonstrating its effectiveness.
