import gradio as gr
import yaml
import json
import requests
import numpy as np
from scipy.special import softmax

FOR_CONTAINER = True
docker_port = 5001
if not FOR_CONTAINER:
    with open('./configs/main.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    docker_port = cfg['DOCKER_PORT']

tweets = [
    "Just had the best day ever! My team won the championship game 🏆🎉",
    "Absolutely love my new coffee mug! It's so cute and comfy 😍",
    "What a beautiful sunset today! Perfect evening walk 🌅🌳",
    
    "Ugh, traffic is terrible today. Stuck in a jam for hours 🚗😡",
    "Can't believe my favorite restaurant closed down. Nowhere else compares 🍴😢",
    "Just spilled coffee all over my new shirt. Ruined my day ☕️😤",
    
    "Interesting article about climate change. Food for thought 📊💭",
    "My cat is sleeping peacefully. Such a sweet creature 🐱😴",
    "Just finished a great book. Highly recommend it! 📖✨"
]

def predict(text=None):
    features = {
        "text": text
    }

    example = json.dumps( 
        { "inputs": features }
    )

    payload = example

    response = requests.post(
        url=f"http://api:{docker_port}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    output_array = np.array(response.json()['prediction'])
    probabilities = softmax(output_array)
    sentiment_index = np.argmax(probabilities)

    sentiments = ['Negative', 'Neutral', 'Positive']
    colors = ['#FF0000', '#FFFF00', '#008000']  # Red, Yellow, Green in HEX

    sentiment = sentiments[sentiment_index]

    max_probability = probabilities.max()
    background_color = colors[sentiment_index]
    html_output = f"<div style='background-color:{background_color}; padding: 10px; border-radius: 10px;'>"
    html_output += f"<h2>{sentiment}: {max_probability:.2%}</h2></div>"

    return html_output

# Only one interface is enough
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Text(label="tweet")
    ],
    outputs=gr.HTML(label="Sentiment"),
    examples=tweets
)

# Launch the web UI locally on port 5155
if __name__ == '__main__':
    demo.launch(share=False, server_name='0.0.0.0', server_port=5155)
