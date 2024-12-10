import boto3
import time

def batch_inference(input_file):
    client = boto3.client('sagemaker')

    job_name = f"bedrock-titan-batch-job-{time.strftime('%Y%m%d%H%M%S')}"
    endpoint_name = "bedrock-titan-proxy"

    client.create_transform_job(
        TransformJobName=job_name,
        ModelName=endpoint_name,
        TransformInput={
            "S3Uri": f"s3://<YOUR_BUCKET_NAME>/{input_file}",
            "ContentType": "application/jsonlines",
        },
        TransformOutput={
            "S3Uri": f"s3://<YOUR_BUCKET_NAME>/results/",
        },
        TransformResources={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
        },
    )

    print(f"Started batch job: {job_name}")
    status = None
    while status not in ("Completed", "Failed", "Stopped"):
        response = client.describe_transform_job(TransformJobName=job_name)
        status = response["TransformJobStatus"]
        print(f"Job status: {status}")
        time.sleep(30)
    
    if status == "Completed":
        print(f"Batch inference completed. Results in S3.")
    else:
        print(f"Batch inference failed. Check SageMaker console for details.")

if __name__ == "__main__":
    batch_inference("test_input.jsonl")
