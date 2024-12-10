import boto3

def cleanup_resources():
    client = boto3.client('sagemaker')

    endpoint_name = "bedrock-titan-proxy"
    config_name = "bedrock-titan-config"
    model_name = "bedrock-titan-proxy"

    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=config_name)
    client.delete_model(ModelName=model_name)

    print(f"Deleted endpoint, config, and model for {endpoint_name}.")

if __name__ == "__main__":
    cleanup_resources()
