import boto3
import json

class BedrockModelProxy:
    def __init__(self, model_name, max_retries=100):
        self.client = boto3.client('bedrock-runtime')  # Amazon Bedrock runtime client
        self.model_name = model_name
        self.max_retries = max_retries

    def invoke(self, payload):
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=json.dumps(payload),
                    contentType="application/json",
                )
                return json.loads(response['body'])
            except self.client.exceptions.ThrottlingException:
                retries += 1
                print(f"ThrottlingException encountered. Retrying {retries}/{self.max_retries}...")
        raise Exception("Max retries reached.")




 