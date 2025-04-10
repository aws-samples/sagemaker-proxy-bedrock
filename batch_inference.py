import boto3
import json
import time
from botocore.exceptions import ClientError

class BatchInferenceManager:
    def __init__(self, region_name='us-east-1'):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region_name)
        self.region = region_name

    def process_batch_inference(self, input_file_path):
        """Process batch inference on local JSON file"""
        try:
            # Read input file
            print(f"Reading input file: {input_file_path}")
            with open(input_file_path, 'r') as f:
                input_data = json.load(f)
            
            # Process each input
            results = []
            total_inputs = len(input_data['input'])
            
            print(f"\nProcessing {total_inputs} inputs...")
            print("=" * 80)
            
            for i, question in enumerate(input_data['input'], 1):
                print(f"\nProcessing input {i}/{total_inputs}")
                
                # Prepare payload
                payload = {
                    "input": [question]
                }
                
                # Invoke endpoint
                try:
                    response = self.sagemaker_runtime.invoke_endpoint(
                        EndpointName='bedrock-titan-proxy',
                        ContentType='application/json',
                        Body=json.dumps(payload)
                    )
                    
                    # Parse response
                    response_body = json.loads(response['Body'].read().decode())
                    results.append(response_body)
                    
                    # Print progress
                    print(f"Input: {question}")
                    print(f"Response: {response_body}")
                    print("-" * 80)
                    
                except ClientError as e:
                    print(f"Error processing input {i}: {str(e)}")
                    results.append({"error": str(e)})
            
            # Print final results
            print("\nFinal Results:")
            print("=" * 80)
            for i, (question, result) in enumerate(zip(input_data['input'], results), 1):
                print(f"\nQuestion {i}:")
                print(f"Input: {question}")
                print(f"Response: {result}")
                print("-" * 80)
            
            return results
            
        except Exception as e:
            print(f"Error in batch inference: {str(e)}")
            raise

def main():
    # Initialize the manager
    manager = BatchInferenceManager()
    
    try:
        print("Starting batch inference process...")
        results = manager.process_batch_inference('test_input.json')
        
        # Save results to file
        output_file = 'inference_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
            
    except Exception as e:
        print(f"Error in batch inference process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
