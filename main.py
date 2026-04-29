import boto3
import json

bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='eu-north-1')

prompt = 'Help me set up networking between two docker containers.'

