# README: Adaptive Focal Loss in Multi-Task Personality Recognition for Mitigating Hard Class Imbalance Bias

## Overview
This project explores class imbalance issues in personality recognition tasks and introduces a novel **Adaptive Focal Loss** (AFL) designed for multi-task learning. The approach leverages advanced linguistic features and custom loss functions to improve classification performance on datasets with skewed personality type distributions.

## Abstract Summary
Personality recognition is pivotal for personalized responses and understanding user behavior. However, class imbalance in existing datasets hampers model performance. This study evaluates various **Class Imbalance Mitigation Techniques** (CIMTs), introduces the **Adaptive Focal Loss**, and recommends balanced evaluation metrics. Experiments were conducted using **Linguistic Inquiry and Word Count (LIWC)** and **Term Frequency-Inverse Document Frequency (TF-IDF)** features. Results highlight significant performance gains through trainable hyperparameters, mitigating the challenges of parameter selection and sensitivity.

## Key Features
- **Adaptive Focal Loss:** A novel loss function that adapts dynamically during training to handle class imbalance. Supports simultaneous personality recognition across multiple tasks.
- **Advanced Feature Engineering:** Utilizes LIWC and TF-IDF features to capture linguistic nuances.
- **Comprehensive Evaluation Metrics:** Balanced accuracy is recommended over regular accuracy and F1 score for imbalanced datasets.

## Dataset Details
Two datasets were utilized in the experiments:
1. [MBTI Kaggle Dataset](https://huggingface.co/datasets/jingjietan/kaggle-mbti): Contains labeled data for Myers-Briggs personality types.
2. [Essays Big Five Dataset](https://huggingface.co/datasets/jingjietan/essays-big5): Includes essay data annotated with Big Five personality traits.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/adaptive-focal-loss.git
   cd adaptive-focal-loss
   ```
2. Create a virtual environment and install dependencies:
   ```bash
    conda create --name afl-env python=3.8
    conda activate afl-env
    pip install -r requirements.txt
   ```

3. **Download the datasets**  
    You can download the datasets manually or use the Hugging Face library.

    - **Using Hugging Face Library**  
        -  **Install the `datasets` library:**
            ```bash
            pip install datasets
            ```
        - **Download the datasets:**
            ```python
            from datasets import load_dataset

            # Load MBTI Kaggle Dataset
            mbti_kaggle = load_dataset('jingjietan/kaggle-mbti')

            # Load Essays Big Five Dataset
            essays_big5 = load_dataset('jingjietan/essays-big5')

            # Save datasets to the `dataset/` directory
            mbti_kaggle.to_csv('dataset/mbti_kaggle.csv')
            essays_big5.to_csv('dataset/essays_big5.csv')
            ```

    - **Download**
        - The datasets used in this project include:
            - [Essays Big5 Dataset](https://huggingface.co/datasets/jingjietan/essays-big5)
            - [Kaggle MBTI Dataset](https://huggingface.co/datasets/jingjietan/kaggle-mbti)


## Usage

### Getting started
1. Change directory to model training:
    ```bash
    cd model_training
    ```
2. Run the script:
    ```bash
    python train.py
    ```


## Results
- **Adaptive Focal Loss** significantly improved classification performance, particularly on underrepresented classes.
- **Trainable Hyperparameters:** The dynamic adaptation of hyperparameters eliminates manual tuning and enhances generalization.
- **Evaluation Recommendation:** Balanced accuracy provides a more reliable measure of performance in imbalanced datasets compared to traditional metrics.


