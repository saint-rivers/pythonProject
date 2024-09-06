from huggingface_hub import hf_hub_download
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

REPO_ID = "saintrivers/test_eli5"

# Create a Flask application
app = Flask('churn')

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    fill_mask = request.get_json()
    text = fill_mask['text']

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = TFAutoModelForMaskedLM.from_pretrained(REPO_ID)

    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0,1]

    logits=model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

    out = []
    for token in top_3_tokens:
        res = text.replace(tokenizer.mask_token, tokenizer.decode([token]))
        out.append(res)

    # Prepare the result in JSON format
    result = {
        'results': out
    }

    # Return the result as a JSON response
    return jsonify(result)

# Run the Flask application
if __name__ == "__main__":
    # Run the app in debug mode on all available network interfaces
    app.run(debug=True, host='0.0.0.0', port=9696)
