from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
import imghdr

# helpers
def detect_mime(data: bytes) -> str:
    # PDF
    if data.startswith(b"%PDF"):
        return "application/pdf"

    # images
    img = imghdr.what(None, h=data)
    if img:
        return f"image/{img}"

    return "application/octet-stream"

# setup
load_dotenv()

app = Flask(__name__)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_KEY)

# routes
@app.get("/")
def health():
    return "backend alive"

@app.post("/analyze")
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        uploaded_file = request.files["file"]

        # read bytes
        file_bytes = uploaded_file.read()

        # detect correct mime
        mime_type = detect_mime(file_bytes)

        print("Detected mime:", mime_type)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=file_bytes,
                                mime_type=mime_type
                            )
                        ),
                        types.Part(
                            text=(
                                "You are a senior technical recruiter.\n"
                                "Return ONLY valid JSON with:\n"
                                "{ overall, strengths[], weaknesses[], improvements[], rewrite_example }"
                            )
                        )
                    ]
                )
            ]
        )

        # safer parsing
        text = response.text or ""
        # remove markdown fences if Gemini adds them
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(text)
            return jsonify(parsed)
        except Exception as e:
            print("Parse failed:", text)
            return jsonify({"error": "Invalid model output"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)