# Regex Pattern Generator

A regular expression pattern generator powered by artificial intelligence

## Table of contents

- [Regex Pattern Generator](#regex-pattern-generator)
  - [Table of contents](#table-of-contents)
  - [Model Overview](#model-overview)
  - [To Do Until v1.0](#to-do-until-v10)

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

- ### **Architecture**

  - Uses LSTM layers with Go-Backwards enabled to create multi-layer RNN - create one for Matching texts and one for Rejected texts
  - Concatenates outputs or RNN's
  - Uses standard dense hidden layers with relu activation
  - Output from a dense layer with tanh activation

- ### **Loss**

  - Uses mean squared error (mse) loss function

- ### **Evaluation**

  - Use a test and evaluation set to evaluate the performance of the model
  - Calculate metrics such as precision, recall, and F1-score to measure the model's accuracy in generating regex statements

## To Do Until v1.0

- ~~Create Model (completed)~~
- Obtain Larger Dataset
- Improve model accuracy
- Ensure model outputs valid regex
- ~~Convert to functional api, not model subclassing (completed)~~
- Create Frontend
