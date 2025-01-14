import os
import json
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import threading

app = Flask(__name__)

# Modellpfade und Konfigurationen
MODEL_PATH = "./fine_tuned_gpt2"
NEW_DATA_PATH = "./new_data.json"
LOCK = threading.Lock()

# Modell initialisieren
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    notes = data.get("notes", "")
    
    # Text generieren
    response = generator(notes, max_length=100, num_return_sequences=1)
    generated_text = response[0]["generated_text"]
    
    # Nutzereingabe und generierter Text speichern
    save_new_data(notes, generated_text)
    return jsonify({"text": generated_text})


def save_new_data(notes, generated_text):
    """
    Neue Daten speichern, um sie für das erneute Training zu verwenden.
    """
    new_entry = {"input": notes, "output": generated_text}
    with LOCK:
        if os.path.exists(NEW_DATA_PATH):
            with open(NEW_DATA_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(new_entry)

        with open(NEW_DATA_PATH, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Neues Training basierend auf gespeicherten Daten starten.
    """
    def train_and_update_model():
        # Hier wird ein neues Training durchgeführt
        os.system("python retrain_model.py")
        
        # Aktualisiere das Modell nach dem Training
        global model, generator
        with LOCK:
            model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Modell erfolgreich aktualisiert!")

    thread = threading.Thread(target=train_and_update_model)
    thread.start()

    return jsonify({"status": "Retraining gestartet"}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
