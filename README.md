# Supplementary Material for "What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach"

This repository contains the supplementary material for the manuscript entitled "What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach". The material provided is intended to support the findings and methodologies discussed in the paper.

## Repository Structure

The repository is organized as follows:

### `src` Folder

- **utils**: Utility functions and helper scripts
  - **dataset.py**: Dataset classes for different log formats
  - **tokenizer.py**: Log sequence tokenization utilities
  - **seq_encoder.py**: Implementations of positional and temporal encoding methods
  - **earlystopping.py**: Early stopping implementation for training
  - **config.py**: Configuration utilities
- **anomaly_model.py**: Defines the main anomaly detection model architecture
- **anomaly_bilstm.py**: Implementation of BiLSTM-based anomaly detection
- **anomaly_neurallog.py**: Implementation of NeuralLog architecture
- **model.py**: Contains the base transformer model definitions
- **train_anomaly_binary.py**: Training script for binary anomaly detection
- **train_neurallog.py**: Training script for NeuralLog model
- **paths.yaml**: Configuration file for different environment paths

## Usage

To use the supplementary material, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/mooselab/suppmaterial-CfgTransAnomalyDetector.git 
   cd ./suppmaterial-CfgTransAnomalyDetector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure paths:
   - Update `paths.yaml` with appropriate paths for your environment
   - Set up dataset paths according to the specified structure

4. Training models:
   - For transformer-based anomaly detection:
     ```bash
     python train_anomaly_binary.py --dataset BGL --env local --seq_enc_method temporal --d_model 64 --embed_method sentence
     ```
   - For NeuralLog-based detection:
     ```bash
     python train_neurallog.py --dataset BGL --env local
     ```

   Available arguments:
   - `--dataset`: Dataset to use (default: 'BGL')
   - `--env`: Environment configuration ('local' or 'cluster')
   - `--seq_enc_method`: Sequence encoding method ('temporal', 'positional', 'time2vec', 'None', 'temporal_only', 'time2vec_only')
   - `--d_model`: Model dimension (default: 64)
   - `--embed_method`: Log embedding method ('sentence', 'random', 'onehot')

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

## Reference

To be available after the review process.
