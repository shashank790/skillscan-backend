from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json

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
        mime_type = uploaded_file.mimetype  # THIS replaces imghdr

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
                                "You are a senior technical recruiter and ATS expert.\n"
                                "You will be given a resume as a PDF or image.\n"
                                "\n"
                                "CRITICAL OUTPUT RULES:\n"
                                "- Return ONLY raw JSON. No markdown, no code fences.\n"
                                "- The JSON MUST parse with json.loads.\n"
                                "- Always include EVERY required key below, even if empty.\n"
                                "- Use concise, actionable language.\n"
                                "\n"
                                "REQUIRED JSON SHAPE:\n"
                                "{\n"
                                '  "score": number,\n'
                                '  "ats_score": number,\n'
                                '  "overall": string,\n'
                                '  "strengths": string[],\n'
                                '  "weaknesses": string[],\n'
                                '  "improvements": string[],\n'
                                '  "rewrite_example": string\n'
                                "}\n"
                                "\n"
                                "FIELD GUIDANCE:\n"
                                "- score: integer 0-100 (overall resume strength for a general\n"
                                "  software/tech role). Use a strict integer.\n"
                                "- ats_score: integer 0-100 (ATS readability). How well the\n"
                                "  resume will parse and rank in typical Applicant Tracking\n"
                                "  Systems: structure, clear headings, no complex formatting,\n"
                                "  keyword presence, scannability. Use a strict integer.\n"
                                "- overall: 2-4 sentences summarizing the resume quality,\n"
                                "  clarity, impact, and ATS readability.\n"
                                "- strengths: 3-6 bullets. Each bullet should mention a\n"
                                "  specific positive (metrics, scope, tech stack, clarity).\n"
                                "- weaknesses: 3-6 bullets. Each bullet should be concrete.\n"
                                "- improvements: 5-8 bullets. Each bullet should be a\n"
                                "  next action the candidate can do today.\n"
                                "- rewrite_example: ONE rewritten bullet. Format:\n"
                                "  'Before: ...\\nAfter: ...'\n"
                                "\n"
                                "If the resume content is unreadable/too blurry, set:\n"
                                '- score: 0, ats_score: 0\n'
                                '- overall: explain it is unreadable\n'
                                "- strengths/weaknesses/improvements: []\n"
                                "- rewrite_example: ''\n"
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
    app.run()