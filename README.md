

# AutoVC on Custom Dataset

## Overview

This repository provides an implementation of **AutoVC**, a **voice conversion model** that enables one-shot voice cloning without requiring parallel data. AutoVC uses an **autoencoder-based architecture** to disentangle speaker identity from speech content, making it ideal for converting one speaker’s voice to another using only a few seconds of target speaker data.

This project adapts AutoVC to work with a **custom dataset**, allowing personalized voice conversion and experimentation via a **Jupyter Notebook**.

## Features

* **Autoencoder-based architecture**: Learns to separate content and speaker identity for voice conversion.
* **One-shot voice cloning**: Clone a target speaker's voice using only a short reference clip.
* **Custom dataset support**: Easily train and test the model using your own **.wav** dataset.
* **Notebook-driven interface**: All steps from preprocessing to inference are handled inside a **Jupyter Notebook**.
* **Checkpoint saving**: Training progress is stored in checkpoint files for continuation or evaluation.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/AutoVC_CustomDataset.git
   cd AutoVC_CustomDataset
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Prepare a folder named `data` at the project root with subfolders for each speaker, each containing **.wav** files sampled at a consistent rate (e.g., 16kHz). Ensure all files are clean and normalized for optimal results.

## Running the Notebook

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook autovc-result.ipynb
   ```

2. Follow the step-by-step sections:

* Data preprocessing and mel-spectrogram generation
* Model training and evaluation
* Voice conversion and synthesis
* Checkpoint loading and saving

## Training

Modify training parameters in the notebook:

* `num_epochs`: Number of epochs for training
* `batch_size`: Training batch size
* `lr`: Learning rate

## Voice Conversion

Once trained, you can convert source audio into the target speaker's voice using short reference audio samples. Output files will be saved in the `output/` directory.

## Model Checkpoints

Training checkpoints are saved in the `checkpoints/` folder. You can resume training or use them for inference at any time.

## Results

Converted voice samples are saved as **.wav** files in the `output/` directory. These can be evaluated by listening or using voice similarity metrics.

---

# SV2TTS on Custom Dataset

## Overview

This repository contains an implementation of **SV2TTS**, a three-stage pipeline for **realistic voice cloning** using speaker embeddings. SV2TTS uses:

* **Speaker Encoder** (to extract speaker embeddings),
* **Synthesizer** (to generate mel spectrograms from text + embeddings), and
* **Vocoder** (to generate waveform audio from mel spectrograms).

This implementation is adapted to work with **custom datasets** and is structured within a **Jupyter Notebook** for easier experimentation and modification.

## Features

* **Three-stage cloning pipeline**: Encoder, synthesizer, and vocoder for high-quality TTS.
* **Few-shot speaker cloning**: Clone voices using just a few seconds of target audio.
* **Custom dataset support**: Train each stage with your own audio data.
* **Interactive Jupyter Notebook workflow**: Modular execution of each component.
* **Checkpoints for each stage**: Save progress and perform inference at any step.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/SV2TTS_CustomDataset.git
   cd SV2TTS_CustomDataset
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

* For the **Speaker Encoder**, you need clean, trimmed **.wav** files per speaker.
* For the **Synthesizer**, you need paired **text-audio** samples (e.g., from audiobooks or TTS corpora).
* For the **Vocoder**, mel spectrograms and their corresponding **.wav** files are required.

Organize your dataset under the `data/` folder, structured appropriately for each stage.

## Running the Notebook

1. Start Jupyter:

   ```bash
   jupyter notebook sv2tts-result.ipynb
   ```

2. Run the notebook step-by-step:

* Speaker embedding training and extraction
* Synthesizer training and inference
* Vocoder training and waveform generation

## Training

Adjust hyperparameters for each stage in the notebook as needed:

* `epochs`
* `batch_size`
* `learning_rate`

Each stage includes options for saving and resuming training from checkpoints.

## Voice Cloning

To clone a voice:

1. Extract speaker embeddings from a reference sample.
2. Provide text input to the synthesizer.
3. Generate audio from the resulting spectrogram using the vocoder.

Outputs are stored in the `output/` directory.

## Model Checkpoints

All checkpoints are saved during training under the `checkpoints/` directory. You can load them independently for each module.

## Results

The final synthesized speech, cloned from a target speaker, is saved as **.wav** files in the `output/` folder. You can test realism and speaker similarity using both subjective and objective evaluation methods.


# WaveGAN on Custom Dataset

## Overview
This repository contains an implementation of **WaveGAN** for generating audio samples based on a custom dataset. **WaveGAN** is a generative adversarial network designed to synthesize 1D waveform data, specifically for audio generation tasks such as sound generation or voice cloning.

This project trains **WaveGAN** using a custom dataset of audio files. The implementation is provided in a **Jupyter Notebook** format, allowing for easy experimentation and modification.

## Features
- **WaveGAN architecture**: A **GAN** framework designed for generating raw audio waveforms.
- **Custom dataset support**: Allows training on your own dataset of **.wav** files.
- **Notebook-based workflow**: The entire process from data preprocessing to audio generation is handled inside a **Jupyter Notebook**.
- **Model checkpoints**: Save and resume training with **checkpoints**.

- ## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Alexalen0/WaveGAN_CustomDataset_Kaggle.git
   cd WaveGAN_CustomDataset_Kaggle

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Dataset
The dataset should consist of .wav files, all sampled at the same rate (e.g., 16kHz). Place your dataset in a folder named data at the project root. Ensure that your audio files are preprocessed and ready for training.

## Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook wavegan-result.ipynb
2. Open the notebook and follow the step-by-step instructions to run the model. The notebook contains sections for:
 - Data loading and preprocessing
 - Model training
 - Audio sample generation
 - Saving and loading model checkpoints

## Training
To train the WaveGAN model on your custom dataset, run the training cells in the notebook. You can adjust hyperparameters such as:

- epochs: Number of epochs to train
- batch_size: Size of the batches for training
- learning_rate: The learning rate for the optimizer

## Generating Audio Samples
- Once training is complete, you can generate new audio samples by running the generation cells in the notebook. Generated audio files will be saved in the output directory.

## Model Checkpoints
Model checkpoints will be automatically saved during training in the checkpoints folder. You can resume training or generate new samples by loading these checkpoints in the notebook.

## Results
The generated audio files will be saved as .wav files in the output folder. You can listen to these samples using any media player that supports the .wav format.
