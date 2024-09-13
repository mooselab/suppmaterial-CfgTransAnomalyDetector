# Supplementary Material for "What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach"

This repository contains the supplementary material for the manuscript entitled "What Information Contributes to Log-based Anomaly Detection? Insights from a Configurable Transformer-Based Approach". The material provided is intended to support the findings and methodologies discussed in the paper.

## Repository Structure

The repository is organized as follows:

### `src` Folder

- **utils**: Utility functions and helper scripts. The implementations of positional and temporal encoding methods are included.
- **anomaly_bilstm.py**: Script for anomaly detection using BiLSTM.
- **anomaly_model.py**: Defines the anomaly detection model architecture.
- **eval_anomaly_bin.py**: Evaluation script for binary anomaly detection.
- **model.py**: Contains the transformer model definitions and configurations.
- **sentence_embedding_generation.py**: Script for generating sentence embeddings.
- **train_anomaly_binary.py**: Training script for binary anomaly detection.


## Usage

To use the supplementary material, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/LogAnalyticsResearcher/CfgTransAnomalyDetector.git
   cd CfgTransAnomalyDetector
   ```
   
2. Install the required dependencies:
```pip install -r requirements.txt```

3. Navigate to the `src` directory and run the desired scripts.
   - First, semantic embeddings for log templates should be generated with **sentence_embedding_generation.py**.
   - Modify the parameters within **train_anomaly_binary.py**.
   - Train and test the model: ``python train_anomaly_binary.py``

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

## Reference

To be available after the review process.
