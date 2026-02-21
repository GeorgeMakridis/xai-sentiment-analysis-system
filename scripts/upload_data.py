import requests
import json

# Upload data to XAI service
def upload_data():
    url = "http://localhost:8000/ingest"
    
    data = {
        "file_path": "/app/shared_data/uploads/news_sentiment_updated.json",
        "user_id": "admin",
        "data_type": "sentiment"
    }
    
    response = requests.post(url, json=data)
    print(f"Upload response: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    success = upload_data()
    if success:
        print("Data uploaded successfully!")
    else:
        print("Failed to upload data") 