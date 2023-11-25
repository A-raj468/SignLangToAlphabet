# Sign Language to Alphabet converter

This project utilizes Convolutional Neural Networks (CNN) and custom data creation to interpret sign language in real-time.

## Table of Contents

- [Sign Language to Alphabet converter](#sign-language-to-alphabet-converter)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Creating the dataset](#creating-the-dataset)
    - [Training the model](#training-the-model)
    - [Using live video feed](#using-live-video-feed)
  - [Libraries Used](#libraries-used)

## Introduction

This project is part of the CS335 (AI/ML Lab) course instructed by Prof. Preethi Jyothi.

Collaboratively developed by:

- Aditya Raj
- Hruday Nandan Tudu
- M Hrushikesh Reddy
- Shikhar Parmar
- Venkata Panees Dugineni

## Installation

To get started with the application, clone the repository and install the necessary dependencies using the `requirements.txt` file.

```bash
# Example installation steps
git clone https://github.com/A-raj468/SignLangToAlphabet.git
cd SignLangToAlphabet
pip install -r requirements.txt
```

## Usage

### Creating the dataset

Begin by creating your dataset using `collect_images.py`.

```bash
python collect_images.py
```

Capture images for each alphabet by pressing 'Q'.

### Training the model

Train the model using `SignLangToAlphabet.py`.

```bash
python SignLangToAlphabet.py
```

### Using live video feed

Run the `inference_classifier.py` to start the live feed. Show sign language with one hand. Quit by presing 'Q'.

## Libraries Used

- PyTorch
- Tqdm
