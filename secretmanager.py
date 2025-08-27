import boto3
import json
import os

def get_secret(secret_name=None, region_name=None):
    if secret_name is None:
        secret_name = os.getenv('SECRET_NAME')
    
    if region_name is None:
        region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
    
    if not secret_name:
        print("Warning: No secret_name provided and SECRET_NAME environment variable not set.")
        return {}
    
    try:
        client = boto3.client('secretsmanager', region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response['SecretString'])
        print(f"Successfully loaded secrets from {secret_name}")
        return secrets
    except Exception as e:
        print(f"Warning: Could not load secrets from AWS Secrets Manager: {e}")
        print("Falling back to environment variables")
        return {}


def get_api_key(key_name, secrets=None, secret_name=None, region_name=None):
    if secrets is None:
        secrets = get_secret(secret_name, region_name)
    
    # Try to get from secrets first, then fallback to environment variables
    api_key = secrets.get(key_name) or os.getenv(key_name)
    
    if not api_key:
        print(f"Warning: {key_name} not found in secrets or environment variables")
    
    return api_key


def load_all_secrets(secret_name=None, region_name=None):
    secrets = get_secret(secret_name, region_name)
    
    # Set environment variables for all secrets
    for key, value in secrets.items():
        if value:  # Only set non-empty values
            os.environ[key] = value
    
    if secrets:
        print(f"Loaded {len(secrets)} secrets as environment variables")
    
    return secrets


def get_google_credentials(secrets):
    try:
        # Get the JSON key from AWS Secrets Manager
        google_creds_json = get_api_key('GOOGLE_CREDENTIALS', secrets)
        if google_creds_json:
            # Write to temporary file or use directly
            import tempfile
            import json
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                if isinstance(google_creds_json, str):
                    f.write(google_creds_json)
                else:
                    json.dump(google_creds_json, f)
                return f.name
    except Exception as e:
        print(f"Could not load Google credentials: {e}")
        return None