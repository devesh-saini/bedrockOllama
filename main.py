import boto3
import json

bedrock_client = boto3.client(service_name='bedrock-agent-runtime', region_name='eu-north-1')

prompt = 'Help me set up networking between two docker containers.'

knowledge_base_retrieval = bedrock_client.retrieve(
    knowledgeBaseId='D9GSQGHM9G',
    retrievalQuery={
        "text": prompt
    }
)

print(knowledge_base_retrieval['retrievalResults'])