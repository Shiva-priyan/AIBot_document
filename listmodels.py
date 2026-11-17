import os, requests
from dotenv import load_dotenv

load_dotenv()
r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                 params={"key": os.getenv("GOOGLE_API_KEY"), "pageSize": 200})
for m in r.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        print(m["name"].replace("models/", ""))

        