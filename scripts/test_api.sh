#!/bin/bash

# Define the URL of the endpoint
URL="http://localhost:8080/predict"

# Define the headers
HEADERS="Content-Type: application/json"

# Define the data payload
DATA='{
    "inputs": {
        "text": "I am sorry to hear about your setback. Do not give up – keep pushing forward."
    }
}'

# Send the POST request
RESPONSE=$(curl -s -X POST "$URL" -H "$HEADERS" -d "$DATA")

# Print the response
echo "$RESPONSE"