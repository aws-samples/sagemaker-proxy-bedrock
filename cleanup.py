import boto3

def cleanup():
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        # Delete the endpoint
        sagemaker_client.delete_endpoint(EndpointName='bedrock-titan-proxy')
        print("Deleting SageMaker endpoint bedrock-titan-proxy...")
        
        # Delete the endpoint configuration
        sagemaker_client.delete_endpoint_config(EndpointConfigName='bedrock-titan-proxy')
        
        # Delete the model
        sagemaker_client.delete_model(ModelName='bedrock-titan-proxy')
        
        print("Done.")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    cleanup()
