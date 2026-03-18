import os

# Set test env vars before any imports
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("STORAGE_PATH", "/tmp/dr-test-output")
os.environ.setdefault("PROJECTS_PATH", "./projects")
