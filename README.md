# Regex Pattern Generator

A regular expression pattern generator powered by artificial intelligence and neural networks.
> *This project is still under development, and may output flawed regular expressions*

## Table of contents

- [Regex Pattern Generator](#regex-pattern-generator)
  - [Table of contents](#table-of-contents)
  - [Model Overview](#model-overview)
  - [Dataset](#dataset)
  - [FAQ](#faq)
    - [What is a regular expression?](#what-is-a-regular-expression)
    - [Why reverse engineer regex?](#why-reverse-engineer-regex)
    - [How can I help contribute?](#how-can-i-help-contribute)
  - [To Do Until v1.0 Release](#to-do-until-v10-release)
  - [Future Plans/Ideas](#future-plansideas)

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

  - RNN made from Bidirectional LSTM layers to create encoder that captures underlying data from the input into a latent space
  - Multi-Headed Attention Layer to discover differences in allowed and rejected encodings
  - Hidden dense layers with relu activation
  - Output from a dense layer with tanh activation

- ### **Loss**

  - Uses mean squared error (mse) loss function
  - See [TODO](#to-do-until-v10-release) to see details about development of loss implmentation

- ### **Evaluation**

  - Use a test and evaluation set to evaluate the performance of the model
  - Calculate metrics such as precision, recall, and F1-score to measure the model's accuracy in generating regex statements

## Dataset

The dataset is not currently large enough for the model to be accurate. We need a few thousand more data entries before the model will be functional. Feel free to add a few examples to the sqlite db and submit a pull request, or if you know a better way to get this data, let me know.

## FAQ

### What is a regular expression?

A regular expression, regex, or regexp is a sequence of characters that specifies a match pattern in a text. Some common uses of regex include text functions (find, replace, split, etc.), input validation, and data extraction.

### Why reverse engineer regex?

Though this project was created for personal education, regex reverse engineering has practical applications. Some examples include:

- **Extraction** - Easily create an extraction pattern from that should be extracted, and data that should not.
- **Offensive Security** - Regex reverse engineering can be a useful tool for ofsec. For example, if you have an input that is filtered or validated by a regular expression, you can use the inputs that pass the validation and inputs that don't to reverse engineer a regex. With knowledge of the regex, you can better find injections that evade the filters.
- **Text Filtering** - You can give inputs that should be filtered, and inputs that should not be filtered, and generate a regex expression to filter texts. DISCLAIMER: This project should not be used any input validation or defensive security applications, as the outputted regex statements are not perfect!

### How can I help contribute?

How can **you** contribute to this project? There's always something that we need help on. Take a look in the [Todo](#to-do-until-v10-release) section to see what needs to be completed before the next release, or take a look at the current project issues. You can also always add your own features - however, it is recommend to open an issue with an "enhancment" tag and seek approval prior to putting any work into a feature. More data is always nice, so if you add a few entries, that would be awesome.

## To Do Until v1.0 Release

- ~~Create Model (completed)~~
- Obtain Larger Dataset (Important)
- Write Custom Loss Function with a combination of
  - ~~Mean Squared Difference Loss (Complete)~~
  - "Is Regex Valid" Loss
  - "Does Regex Match Inputs" Loss
- Add Integer Embedding Layer
- Create Frontend Website (In Progress)
- Create Backend API (In Progress)
- Create Backend Server (Funding or Host Required)

## Future Plans/Ideas

- Implement Transformers?
- Add optional data input parameters - ex. regex length, parts of known regex, regex complexity, etc.
- Far Future: Add natural language processing so a prompt can generate regex from instructions
