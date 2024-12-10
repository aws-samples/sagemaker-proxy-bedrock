import boto3

def deploy_sagemaker_endpoint():
    client = boto3.client('sagemaker')

    model_name = "bedrock-titan-proxy"
    endpoint_config_name = "bedrock-titan-config"
    endpoint_name = "bedrock-titan-proxy"

    # Create model
    client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": "ECR-IMAGE",
            "Environment": {"MODEL_NAME": "amazon.titan-express"}
        },
        ExecutionRoleArn="<YOUR_ROLE_ARN>",
    )

    # Create endpoint configuration
    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "VariantName": "default",
            "ModelName": model_name,
            "InstanceType": "ml.t2.medium",
            "InitialInstanceCount": 1,
        }]
    )

    # Create endpoint
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )
    print(f"Endpoint {endpoint_name} is being created. This might take a few minutes.")

if __name__ == "__main__":
    deploy_sagemaker_endpoint()
