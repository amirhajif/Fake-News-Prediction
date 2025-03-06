# Fake News Prediction

This project aims to predict fake news using machine learning techniques. The model is trained on a dataset of news articles and uses text processing and logistic regression to classify news as real or fake.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Fake news is a significant problem in today's digital age, where misinformation can spread rapidly. This project uses natural language processing (NLP) and machine learning to identify and classify fake news articles.

## Dataset

The dataset used in this project consists of news articles with labels indicating whether they are real or fake. The dataset is preprocessed to remove stopwords and apply stemming before being vectorized using TF-IDF.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/amirhajif/Fake-News-Prediction.git
   cd Fake-News-Prediction
   ```

2. Download the dataset and place it in the project directory.

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook main.ipynb
   ```

4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Model

The model used in this project is a Logistic Regression classifier. The text data is vectorized using TF-IDF, and the model is trained to distinguish between real and fake news articles.

## Results

The model achieves an accuracy of approximately 97.91% on the test set, indicating its effectiveness in predicting fake news.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## Contact

For any questions or inquiries, please contact me at [amirhajitabar2@gmail.com](mailto:amirhajitabar2@gmail.com).

GitHub: [amirhajif](https://github.com/amirhajif)
