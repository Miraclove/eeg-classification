# EEG Classification for Emotion Recognition Using Machine Learning

This project focuses on classifying Electroencephalogram (EEG) signals into different emotional states using machine learning techniques. By leveraging the SEED-IV dataset, we aim to build models capable of distinguishing four emotions: happy, sad, neutral, and fear. The approach includes data pre-processing, feature extraction, and model development using deep learning methods such as CNN and KNN.

## Project Structure

```
.
├── data/                   # Directory for storing datasets and processed data
│   └── preprocessing.ipynb # Jupyter Notebook for data analysis and preprocessing
├── exp/                    # Directory for experimental scripts and training pipelines
│   └── trainer.py          # Script defining the general training loop
├── model/                  # Directory for model architecture definitions
│   ├── ccnn.py             # Python file defining the Convolutional Neural Network model structure
│   └── tscnn.py            # Python file defining the Time-Series Convolutional Neural Network model
├── quick-start.ipynb       # Jupyter Notebook for quick-start guide and code walkthrough
├── README.md               # Project documentation file
├── vanilla_train.py        # Script for training a basic neural network model
└── other scripts/          # Additional training and experiment-related scripts
```

## Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

## Overview

This project is designed to facilitate training, evaluating, and visualizing deep learning models. The project includes various components:
- **Model Structures**: Defined in `model/` directory for easy customization.
- **Data Preprocessing**: Handled through the `data/preprocessing.ipynb` notebook.
- **Training Framework**: Managed by scripts in the `exp/` directory to ensure generality and reusability.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the required dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` or `torch` (depending on model requirements)
- `jupyter`

You can install all the necessary packages by running:
```bash
pip install -r requirements.txt
```

### Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Miraclove/eeg-classification
cd eeg-classification
```

## Data Preprocessing

The data analysis and preprocessing steps are detailed in `data/preprocessing.ipynb`. This notebook includes:
- Loading and inspecting the dataset.
- Visualizing key data features.
- Data cleaning and transformation steps.

To start preprocessing, open the notebook and run the cells step-by-step:
```bash
jupyter notebook data/preprocessing.ipynb
```

## Model Architecture

### Convolutional Neural Network (CNN)
Defined in `model/ccnn.py`, this script includes the structure for a standard CNN used for image-based or structured data classification.

### Temporal-Spatial Convolutional Neural Network (TSCNN)
Defined in `model/tscnn.py`, this model is tailored for Temporal-Spatial data processing.

Both files are organized to make modifications straightforward if different layers or architectures are needed.

## Training the Model

General training routines are encapsulated in `exp/trainer.py`, which offers flexibility to:
- Load different models.
- Adjust training parameters.
- Monitor training performance.

Additionally, you can use the `vanilla_train.py` script for a basic training run:
```bash
python vanilla_train.py
```

## Evaluation

The evaluation metrics and validation steps are embedded in the `trainer.py` script, which reports:
- Training and validation accuracy.
- Loss graphs.
- Optional test set evaluation.

## Future Improvements

- **Hyperparameter Tuning**: Integrate automated hyperparameter search methods like `Optuna` or `Hyperopt`.
- **Interactive Dashboards**: Add dashboards for real-time tracking of training metrics using `TensorBoard` .

## Contributors

- [Weizhi Peng](https://github.com/Miraclove) - Project creator and primary developer.
