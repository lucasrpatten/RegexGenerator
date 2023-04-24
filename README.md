# Regex Pattern Generator

A regular expression pattern generator powered by artificial intelligence

## Table of contents

- [Regex Pattern Generator](#regex-pattern-generator)
  - [Table of contents](#table-of-contents)
  - [Model Overview](#model-overview)
  - [To Do](#to-do)


## Model Overview

- ### **Task**

  - Create an AI that can generate a regex statement for a user

- ### **Input Data**

  - **x**: A list of texts that should match the pattern and a list of texts that should not match
  - **y**: The output regex
  - Varying input length

- ### **Text Preprocessing**

  - Convert each input character to its ascii integer representation. Then divide each integer by 128 to convert the character to a float in the range [0, 1]
  - Pad the inputs so length is consistent

- ### **Encoder**

  - Uses masking to allow for variable sized input regex data
  - Uses Bidirectional LSTM layers to create an RNN
  - Encodes into a latent space that captures underlying patterns in the data

- ### **Sampling**

  - Use a sampling layer with the reparameterization trick to enable backpropogation and stochastic gradient descent

- ### **Decoder**

  - Predicts an output regex from a given latent space generated by the encoder
  - Uses a recurrent network of LSTM layers

- ### **Variational Autoencoder (VAE)**

  - Encodes data, Samples, and Decodes latent space

- ### **Loss**

  - Use a combination of reconstruction loss and KL divergence loss as the VAE loss function

- ### **Evaluation**

  - Use a test set to evaluate the performance of the model
  - Calculate metrics such as precision, recall, and F1-score to measure the model's accuracy in generating regex statements

## To Do

- ~~Create Model (completed)~~
- Create Large Dataset
- Convert to functional api, not subclassing
- Create Frontend
