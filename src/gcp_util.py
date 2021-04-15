"""Google Cloud Utilities

This module contains utilities to access and
manipulate Google Cloud resources.
"""

import os
from google.cloud import storage
from google.oauth2.service_account import Credentials


def get_service_credentials():
    """Authenticate Google service account.

    Uses a credential file to authenticate a google service account.
    It uses the environment variable `GOOGLE_APPLICATION_CREDENTIALS`
    to access the credential file location.

    Returns
    -------
    credential : google.oauth2.service_account.Credentials
    """
    credential_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    credential = Credentials.from_service_account_file(credential_file)
    return credential


def get_bucket():
    """Connect to a valid Google Storage Bucket

    Creates a connection and returns a valid bucket.
    The bucket name must be set as `GOOGLE_BUCKET_NAME`
    environment variable.

    Returns
    -------
    bucket : google.cloud.storage.bucket.Bucket
    """
    env = os.getenv('BERTOLOGY_ENV')
    assert env in ('LOCAL', 'CLOUD', 'COLAB')

    name = os.getenv('GOOGLE_BUCKET_NAME')
    assert name is not None

    if env == 'LOCAL':
        credential = get_service_credentials()
        storage_client = storage.Client(credentials=credential)
        bucket = storage_client.get_bucket(name)
        return bucket

    if env == 'CLOUD':
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(name)
        return bucket

    if env == 'COLAB':
        raise NotImplementedError
