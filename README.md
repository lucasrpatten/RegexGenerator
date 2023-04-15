# Regex Pattern Generator

A regular expression pattern generator powered by artificial intelligence

## Model Overview

- ### Task

  - Create an AI that can generate a regex statement for a user

- ### Training Data

  - A list of texts that should match, a list of texts that should not match, and the expected output. The input length will vary

- ### Recurrent Neural Network (RNN) with attention mechanism

  - Allows for variable sized input regex data
  - Uses attention to focus on relevant parts of the input data
  - Includes length regularization to prevent overfitting

- ### Variational Autoencoder (VAE)

  - Takes the output of the RNN as input
  - Predicts the output regex
  - Includes a latent space to capture underlying patterns in the data

- ### Training

  - Use the training data to train the RNN and VAE simultaneously
  - The RNN is trained to predict the output regex, while the VAE is trained to learn the underlying patterns in the data
  - Use a combination of reconstruction loss and KL divergence loss as the VAE loss function
  - Use backpropagation and stochastic gradient descent to optimize the model weights

- ### Evaluation

  - Use a test set to evaluate the performance of the model
  - Calculate metrics such as precision, recall, and F1-score to measure the model's accuracy in generating regex statements