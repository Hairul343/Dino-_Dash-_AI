# Dino_Dash_AI: An AI-Powered Chrome Dino Game Bot

Welcome to Dino_Dash_AI! This project showcases an AI-powered bot that plays the popular Chrome Dino Game. Leveraging PyTorch and EfficientNet, this bot can automatically navigate the game and avoid obstacles with precision.

![image](https://github.com/Hairul343/Dino-_Dash-_AI/assets/140678940/56acfc08-20dd-435e-9614-cc9a82663318)


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Dino_Dash_AI uses a deep learning model trained with PyTorch to recognize obstacles in the Chrome Dino Game and make real-time decisions about when to jump. The project demonstrates the power of AI in gaming and provides insights into the training and implementation of an AI game bot.

## Features

- **EfficientNet v2_s Architecture**: High accuracy with a small number of parameters.
- **Real-Time Decision Making**: The bot makes decisions on the fly, ensuring smooth gameplay.
- **Data Augmentation**: Enhances training data to improve model generalization.
- **Detailed Documentation**: Comprehensive blog post explaining the training process, model architecture, and bot implementation.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- OpenCV
- NumPy

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Hairul343/Dino_Dash_AI.git
    cd Dino_Dash_AI
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Bot

1. **Start the Chrome Dino Game**:
    Open the Chrome browser and go to `chrome://dino`.

2. **Run the bot**:
    ```bash
    python dino_bot.py
    ```

The bot will start playing the game automatically.

### Training the Model

To train the model from scratch:

1. **Prepare the Dataset**:
    Collect images from the Chrome Dino Game, labeling obstacles and actions (jump or no jump).

2. **Train the Model**:
    ```bash
    python train_model.py --data_path ./data --epochs 50 --batch_size 32
    ```

3. **Evaluate the Model**:
    ```bash
    python evaluate_model.py --model_path ./models/efficientnet_v2_s.pth --test_data_path ./test_data
    ```

## Implementation Details

### Model Architecture

The bot uses the EfficientNet v2_s architecture, chosen for its balance between accuracy and computational efficiency. The model is trained to recognize obstacles and decide when to jump.

### Data Augmentation

Data augmentation techniques such as random cropping, flipping, and color jittering are used to increase the diversity of the training data and improve model generalization.

### Real-Time Decision Making

The bot captures the game screen in real-time, processes the image to detect obstacles, and sends jump commands based on the model's predictions.

## Results

The bot demonstrates high accuracy in recognizing obstacles and making timely jumps, effectively navigating the Chrome Dino Game. Detailed performance metrics and results are available in the blog post linked below.

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository and submit a pull request with your changes. Ensure all tests pass before submitting your PR.

### Guidelines

- Follow the existing code style.
- Write clear and descriptive commit messages.
- Include tests for any new features or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of [PyTorch](https://pytorch.org/) and [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) for their powerful tools.
- Special thanks to the contributors of the [Chrome Dino Game](https://chromedino.com/) for the fun and challenging game.
- For more details on the project, check out the [detailed blog post](https://your-blog-post-link.com).

For any questions or issues, please open an issue on this repository or contact the maintainers.
