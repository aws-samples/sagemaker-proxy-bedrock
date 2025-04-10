import boto3
import time
import json
import os
from botocore.exceptions import ClientError

class BatchInferenceManager:
    def __init__(self, region_name='us-east-1'):
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.region = region_name
        self.input_bucket = 'abulele-sagemaker-proxy-950825-input'
        self.output_bucket = 'abulele-sagemaker-proxy-950825-output'

    def create_bucket_if_not_exists(self, bucket_name):
        """Create an S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} already exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.region
                            }
                        )
                    print(f"Created bucket: {bucket_name}")
                except ClientError as create_error:
                    print(f"Error creating bucket: {str(create_error)}")
                    raise
            else:
                print(f"Error checking bucket: {str(e)}")
                raise

    def upload_file_to_s3(self, file_path, bucket, s3_key):
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(file_path, bucket, s3_key)
            print(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
            return f"s3://{bucket}/{s3_key}"
        except ClientError as e:
            print(f"Error uploading file to S3: {str(e)}")
            raise

    def create_test_input_file(self):
        """Create a test input file"""
        test_data = [
            {
                "input": ["What is the Answer to the Ultimate Question of Life, the Universe, and Everything?"]
            },
            {
                "input": ["What is AWS?"]
            },
            {
                "input": ["Explain machine learning in simple terms."]
            }
        ]
        
        file_path = 'test_input.json'
        with open(file_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        return file_path

    def run_batch_inference(self, input_file_path):
        """Run batch inference job"""
        try:
            # Create buckets if they don't exist
            self.create_bucket_if_not_exists(self.input_bucket)
            self.create_bucket_if_not_exists(self.output_bucket)

            # Upload input file to S3
            input_key = 'input/test_input.json'
            s3_input_path = self.upload_file_to_s3(
                input_file_path, 
                self.input_bucket, 
                input_key
            )

            # Create transform job
            job_name = f"bedrock-titan-batch-job-{time.strftime('%Y%m%d%H%M%S')}"
            
            self.sagemaker_client.create_transform_job(
                TransformJobName=job_name,
                ModelName='bedrock-titan-proxy',
                TransformInput={
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": s3_input_path
                        }
                    },
                    "ContentType": "application/jsonlines",
                    "SplitType": "Line"
                },
                TransformOutput={
                    "S3OutputPath": f"s3://{self.output_bucket}/output/",
                    "Accept": "application/jsonlines"
                },
                TransformResources={
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1
                }
            )

            print(f"Started batch transform job: {job_name}")
            
            # Wait for job completion
            while True:
                response = self.sagemaker_client.describe_transform_job(
                    TransformJobName=job_name
                )
                status = response['TransformJobStatus']
                
                print(f"Job status: {status}")
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    break
                    
                time.sleep(30)
            
            if status == 'Completed':
                print(f"Batch inference completed. Results in s3://{self.output_bucket}/output/")
                return True
            else:
                print(f"Batch inference failed. Status: {status}")
                return False

        except ClientError as e:
            print(f"Error in batch inference: {str(e)}")
            raise

def main():
    # Initialize the manager
    manager = BatchInferenceManager()
    
    try:
        # Create and upload test input file
        input_file = manager.create_test_input_file()
        
        # Run batch inference
        success = manager.run_batch_inference(input_file)
        
        # Cleanup local file
        if os.path.exists(input_file):
            os.remove(input_file)
            
        if success:
            print("Batch inference process completed successfully")
        else:
            print("Batch inference process failed")
            
    except Exception as e:
        print(f"Error in batch inference process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
