
# Cats vs Dogs Classification

This project aims to classify images of cats and dogs using machine learning techniques. The notebook provides a comprehensive walkthrough of the process, including data extraction, preprocessing, model training, and evaluation.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Results](#results)

## Installation

To run this project, you need to have Python and the necessary libraries installed. 

## Usage

To use this project, follow these steps:


1. Extract the datasets (assuming you have the zip files in the `data/input` directory):
    ```bash
    # Extract the training data
    unzip data/input/dogs-vs-cats/train.zip -d data/working/train

    # Extract the test data
    unzip data/input/dogs-vs-cats/test1.zip -d data/working/test
    ```

2. Run the Jupyter notebook:
    ```bash
    jupyter notebook Cats_vs_Dogs.ipynb
    ```

## Project Structure

The project structure is as follows:

```
Cats_vs_Dogs/
├── data/
│   ├── input/
│   │   ├── dogs-vs-cats/
│   │   │   ├── train.zip
│   │   │   ├── test1.zip
│   ├── working/
│       ├── train/
│       │   ├── train/
│       ├── test/
│           ├── test1/
├── Cats_vs_Dogs.ipynb
├── requirements.txt
└── README.md
```

## Data

The data used in this project is sourced from the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data). Ensure you have downloaded and placed the dataset in the `data/input` directory.

## Model

The project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is built using TensorFlow and Keras libraries.

### Steps:
1. **Importing Required Libraries**: Libraries like NumPy, Pandas, OpenCV, TensorFlow, and Keras are imported.
2. **Extracting the Datasets**: The dataset zip files are extracted to the working directory.
3. **Setting up Paths**: Paths to the training and test directories are set up.
4. **Loading and Preprocessing Images**: Functions are defined to load and preprocess images, resizing them to 64x64 pixels.
5. **Visualizing Images**: A function to visualize images along with their labels is provided.

## Results

The results of the model are evaluated using accuracy, precision, recall, and F1 score. Visualization of the training and validation loss and accuracy is also provided.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License.

## Dataset Description

The training archive contains 25,000 images of dogs and cats. Train your algorithm on these files and predict the labels for test1.zip (1 = dog, 0 = cat).

### A note on hand labeling
Per the rules and spirit of this contest, please do not manually label your submissions. We work hard to fair and fun contests, and ask for the same respect in return.
