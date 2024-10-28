import requests
import logging
import os

# Set logging level to DEBUG to capture detailed output
logging.basicConfig(level=logging.DEBUG)

# Ensure Python bypasses any proxy settings for localhost
os.environ['NO_PROXY'] = '127.0.0.1'

# Ollama API URL
url = "http://127.0.0.1:11434/api/generate"

# Set headers for the request
headers = {
    "Content-Type": "application/json"
}

# The data that will be sent in the body of the POST request (JSON format)
data = {
    "model": "llama3.1",
    "prompt": "Generate a match report for Ball Possession: Home 60%, Away 40%. Goals: Home 2, Away 1."
}

# Try making the POST request and catching any exceptions
try:
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful (HTTP 200 OK)
    if response.status_code == 200:
        # Print the raw response content before parsing it as JSON
        print("Raw response content:")
        print(response.content)  # This prints the raw response as bytes
        
        # If it's JSON, try to decode it and handle potential errors
        try:
            response_json = response.json()  # Parse response as JSON
            print("Response JSON:", response_json)  # Print the parsed JSON
            print("Generated Report:", response_json.get("text", "No report generated"))
        except ValueError as e:
            # Handle JSON decoding error
            print(f"JSON decoding error: {e}")
    else:
        # Print the error if the request was not successful
        print(f"Error: {response.status_code} - {response.text}")

except requests.exceptions.RequestException as e:
    # Handle any connection errors (like connection refused)
    print(f"Connection error: {e}")

