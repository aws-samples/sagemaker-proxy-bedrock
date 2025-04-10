import os
import boto3
from sagemaker.pytorch import PyTorchModel
import sagemaker
import torch
import tarfile
import shutil

def deploy_endpoint():
    # Initialize AWS clients
    sagemaker_session = sagemaker.Session()
    
    # Specify your role ARN directly
    role_arn = "YOUR ROLE ARN"
    
    # Create directories for model artifacts
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a dummy PyTorch model file
    dummy_model = torch.nn.Module()
    model_path = os.path.join(model_dir, 'model.pth')
    torch.save(dummy_model.state_dict(), model_path)
    
    # Create tar.gz file
    tar_path = os.path.join(model_dir, 'model.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))
    
    # Package the model files
    model_data = sagemaker_session.upload_data(
        path=tar_path,
        bucket=sagemaker_session.default_bucket(),
        key_prefix='bedrock-proxy/model'
    )

    # Create PyTorch model
    model = PyTorchModel(
        model_data=model_data,
        role=role_arn,  # Using role ARN directly
        entry_point='bedrock_model.py',
        source_dir='.',
        framework_version='2.0.1',
        py_version='py310',
        env={
            'AWS_DEFAULT_REGION': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        }
    )

    try:
        # Deploy the model
        print("Deploying SageMaker endpoint bedrock-titan-proxy...")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name='bedrock-titan-proxy'
        )
        print("Endpoint deployment completed successfully")
        return predictor
    except Exception as e:
        print(f"Error deploying endpoint: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

if __name__ == "__main__":
    deploy_endpoint()
