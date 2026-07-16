import traceback
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import Config
import ee

sa_cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
if sa_cred_path:
    del os.environ['GOOGLE_APPLICATION_CREDENTIALS']

try:
    print("Trying ee.Initialize()...")
    ee.Initialize(project=Config.PROJECT_ID)
    print("Success!")
except Exception as e:
    print("Failed!")
    traceback.print_exc()
