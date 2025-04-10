# ollama_utils.py
import requests
import base64
import tempfile


class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", model="llava"):
        self.base_url = base_url
        self.model = model

    def analyze_image(self, image, prompt=None):
        prompt = prompt or "Ignore the layout or visuals. Just extract all visible text from this image and return it as plain text."


        # Salva imagem temporariamente
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            image.convert("RGB").save(temp, format="JPEG")
            temp_path = temp.name

        # Codifica a imagem em base64
        with open(temp_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        # Prepara payload para /api/chat
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "images": [b64_image],
            "stream": False
        }

        response = requests.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
