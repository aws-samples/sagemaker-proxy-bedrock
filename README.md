# Amazon SageMaker Proxy for Amazon Bedrock

This project demonstrates how to create a SageMaker-compatible dummy model that delegates inference requests to an Amazon Bedrock model. It provides a bridge between SageMaker's robust hosting capabilities and Bedrock's powerful foundation models.

## Files

- `sagemaker_proxy_bedrock/bedrock_model.py`: Contains the dummy model implementation.
- `deploy_endpoint.py`: Script to deploy the SageMaker endpoint.
- `batch_inference.py`: Script to run batch inference using the deployed endpoint.
- `cleanup.py`: Script to clean up the deployed resources.
- `test_input.jsonl`: Sample input file for batch inference.

## Usage

1. Deploy the endpoint:

   ```
   python deploy_endpoint.py
   ```

2. Run batch inference:

   ```
   python batch_inference.py test_input.json
   ```

3. Clean up resources:
   ```
   python cleanup.py
   ```

## Authors

- Ivan Khvostishkov
- Abulele Mditshwa

# Note

Amazon Q Developer in this project.
