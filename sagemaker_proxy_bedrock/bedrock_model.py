import os
import json
import boto3
import time
from typing import List, Dict, Any

def model_fn(model_dir):
    """Load the model for inference"""
    return BedrockModel()

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data.get('input', [])
    raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(data, model):
    """Make prediction"""
    return model.predict(data)

def output_fn(prediction, response_content_type):
    """Format output data"""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction})
    raise ValueError(f'Unsupported content type: {response_content_type}')

class BedrockModel:
    def __init__(self, model_id: str = "amazon.titan-text-express-v1", max_attempts: int = 100):
        self.bedrock_runtime = boto3.client('bedrock-runtime')
        self.model_id = model_id
        self.max_attempts = max_attempts

    def predict(self, inputs: List[str]) -> List[str]:
        responses = []
        for input_text in inputs:
            response = self._invoke_with_retry(input_text)
            responses.append(response)
        return responses

    def _invoke_with_retry(self, input_text: str) -> str:
        attempt = 0
        while attempt < self.max_attempts:
            try:
                request_body = {
                    "inputText": input_text,
                    "textGenerationConfig": {
                        "maxTokenCount": 512,
                        "temperature": 0.7,
                        "topP": 0.9
                    }
                }
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['results'][0]['outputText']
                
            except self.bedrock_runtime.exceptions.ThrottlingException:
                attempt += 1
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** attempt, 60))  # Exponential backoff
                else:
                    raise Exception(f"Max retry attempts ({self.max_attempts}) reached")
            except Exception as e:
                raise e
