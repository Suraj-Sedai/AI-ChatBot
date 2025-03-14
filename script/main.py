from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Pretrained conversational model
model_name = "microsoft/DialoGPT-small"
## Loads corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Loads GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(user_input, chat_history=[]):
    # Tokenize input text
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append input to chat history
    chat_history.append(new_input_ids)

    # Concatenate past conversation
    input_ids = torch.cat(chat_history, dim=-1)

    # Generate response
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode response
    bot_response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return bot_response, chat_history

chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    bot_response, chat_history = chatbot_response(user_input, chat_history)
    print(f"Bot: {bot_response}")

from flask import Flask, request, jsonify

app = Flask(__name__)
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    bot_response, chat_history = chatbot_response(user_input, chat_history)
    
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
