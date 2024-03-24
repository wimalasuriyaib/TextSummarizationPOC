from flask import Flask, request, jsonify
import boto3
import json

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

app = Flask(__name__)

# Call Titan model for text summarization
def summarize_text(text):
    prompt = f"Summarize the following text in 20 words or less: {text}"
    prompt_config = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 20,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 1,
        },
    }

    body = json.dumps(prompt_config)
    modelId = "amazon.titan-text-express-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    summary = response_body.get("results")[0].get("outputText")
    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    summary = summarize_text(text)
    return jsonify({'summary': summary})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8001)
