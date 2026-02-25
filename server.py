from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
import requests
import re
from urllib.parse import urlparse

# setup
load_dotenv()

app = Flask(__name__)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_KEY)

ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.environ.get("ADZUNA_COUNTRY", "us")

GENERIC_ROLE_FALLBACKS = [ # if resume didnt correspond with any titles
    "business analyst",
    "project coordinator",
    "operations analyst",
    "program coordinator",
    "research assistant",
    "customer success manager",
]

# routes
@app.get("/")
def health():
    return "backend alive"


def _strip_fences(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def _repair_json_with_model(bad_json_text):
    repair = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Convert the following into strictly valid JSON.\n"
            "Requirements:\n"
            "- Return ONLY raw JSON.\n"
            "- Preserve all fields and intent.\n"
            "- Escape any inner quotes inside string values.\n"
            "- Do not add markdown/code fences.\n\n"
            f"{bad_json_text}"
        ),
    )
    repaired_text = _strip_fences(repair.text or "")
    return json.loads(repaired_text)

@app.post("/analyze")
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        uploaded_file = request.files["file"]

        # read bytes
        file_bytes = uploaded_file.read()
        mime_type = uploaded_file.mimetype   # THIS replaces imghdr
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
                                "- Do not use unescaped double quotes inside string values.\n"
                                "- Use concise, actionable language.\n"
                                "\n"
                                "REQUIRED JSON SHAPE:\n"
                                "{\n"
                                '  "score": number,                 \n'
                                '  "ats_score": number,             \n'
                                '  "overall": string,               \n'
                                '  "strengths": string[],           \n'
                                '  "weaknesses": string[],          \n'
                                '  "improvements": string[],        \n'
                                '  "rewrite_example": string,       \n'
                                '  "target_roles": string[],        \n'
                                '  "role_domain": string,           \n'
                                '  "role_keywords": string[]        \n'
                                "}\n"
                                "\n"
                                "FIELD GUIDANCE:\n"
                                "- score: integer 0-100 (overall resume strength for the\n"
                                "  candidate's most relevant career track). Use a strict integer.\n"
                                "- ats_score: integer 0-100 (ATS readability). How well the\n"
                                "  resume will parse and rank in typical Applicant Tracking\n"
                                "  Systems: structure, clear headings, no complex formatting,\n"
                                "  keyword presence, scannability. Use a strict integer.\n"
                                "- overall: 2-4 sentences summarizing the resume quality,\n"
                                "  clarity, impact, and ATS readability.\n"
                                "- strengths: 3-6 bullets. Each bullet should mention a\n"
                                "  specific positive (metrics, scope, tech stack, clarity).\n"
                                "- weaknesses: 3-6 bullets. Each bullet should be concrete,\n"
                                "  not generic (e.g., 'Missing measurable impact on projects').\n"
                                "- improvements: 5-8 bullets. Each bullet should be a\n"
                                "  next action the candidate can do today.\n"
                                "- rewrite_example: Provide ONE rewritten bullet demonstrating\n"
                                "  better impact. Format:\n"
                                "  'Before: ...\\nAfter: ...'\n"
                                "- target_roles: 3-5 realistic roles for THIS candidate's actual background.\n"
                                "  Do not force software/tech roles unless the resume supports it.\n"
                                "- role_domain: short category label (e.g. research, education,\n"
                                "  operations, ministry, healthcare, business).\n"
                                "- role_keywords: 5-10 concise job-search keywords reflecting this\n"
                                "  candidate's domain, skills, and seniority.\n"
                                "\n"
                                "If the resume content is unreadable/too blurry, set:\n"
                                '- score: 0, ats_score: 0\n'
                                '- overall: explain it is unreadable\n'
                                "- strengths/weaknesses/improvements: [] (empty arrays)\n"
                                "- rewrite_example: '' (empty string)\n"
                                "- target_roles: []\n"
                                "- role_domain: ''\n"
                                "- role_keywords: []\n"
                            )
                        )
                    ]
                )
            ]
        )

        # safer parsing
        text = _strip_fences(response.text or "")
        try:
            parsed = json.loads(text)
            return jsonify(parsed)
        except Exception as e:
            print("Parse failed:", text)
            try:
                repaired = _repair_json_with_model(text)
                return jsonify(repaired)
            except Exception as repair_error:
                print("Repair failed:", repair_error)
                return jsonify({"error": "Invalid model output"}), 502

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

def _normalize_adzuna_item(item):
    apply_url = item.get("redirect_url")
    if not apply_url:
        return None

    company = item.get("company") or {}
    location = item.get("location") or {}

    return {
        "id": str(item.get("id") or item.get("adref") or apply_url),
        "title": item.get("title") or "",
        "company": company.get("display_name") or "Adzuna",
        "location": location.get("display_name") or "",
        "applyUrl": apply_url,
        "source": "adzuna",
        "postedAt": item.get("created"),
        "summary": item.get("description") or "",
        "salaryMin": item.get("salary_min"),
        "salaryMax": item.get("salary_max"),
        "logoUrl": None,
    }

def _normalize_text_tokens(text):
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return [token for token in normalized.split(" ") if len(token) >= 3]


def _build_adzuna_queries(raw_query, target_roles, role_keywords):
    query = (raw_query or "").strip()
    candidates = []

    if query:
        candidates.append(query[:120])

    if isinstance(target_roles, list):
        for role in target_roles:
            if not isinstance(role, str):
                continue
            cleaned = role.strip().lower()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned[:80])

    keyword_tokens = []
    if isinstance(role_keywords, list):
        for keyword in role_keywords:
            if isinstance(keyword, str):
                keyword_tokens.extend(_normalize_text_tokens(keyword))

    keyword_tokens = list(dict.fromkeys(keyword_tokens))

    if keyword_tokens:
        for role in candidates[:3]:
            combo = f"{role} {' '.join(keyword_tokens[:3])}".strip()
            if combo not in candidates:
                candidates.append(combo[:100])

    for fallback in GENERIC_ROLE_FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)

    # Keep this bounded to avoid excessive provider calls.
    return candidates[:8]


def _score_job(job, target_roles, role_keywords):
    title = (job.get("title") or "").lower()
    summary = (job.get("summary") or "").lower()
    haystack = f"{title} {summary}"
    score = 0

    if isinstance(target_roles, list):
        for role in target_roles:
            if not isinstance(role, str):
                continue
            role_l = role.lower().strip()
            if not role_l:
                continue
            if role_l in title:
                score += 5
            elif role_l in haystack:
                score += 2

    if isinstance(role_keywords, list):
        for keyword in role_keywords:
            if not isinstance(keyword, str):
                continue
            keyword_l = keyword.lower().strip()
            if len(keyword_l) < 3:
                continue
            if keyword_l in haystack:
                score += 1

    return score


def _lookup_logo_for_company(company_name, cache):
    key = (company_name or "").strip().lower()
    if not key:
        return None
    if key in cache:
        return cache[key]

    def _clean_name(name):
        cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
        cleaned = re.sub(
            r"\b(inc|llc|ltd|corp|corporation|co|company|group|holdings|technologies|technology|systems)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _score_candidate(query_name, candidate_name):
        q_tokens = set(_normalize_text_tokens(query_name))
        c_tokens = set(_normalize_text_tokens(candidate_name))
        if not q_tokens or not c_tokens:
            return 0
        return len(q_tokens.intersection(c_tokens))

    queries = [company_name]
    cleaned = _clean_name(company_name)
    if cleaned and cleaned.lower() != company_name.lower():
        queries.append(cleaned)

    try:
        for query in queries:
            response = requests.get(
                "https://autocomplete.clearbit.com/v1/companies/suggest",
                params={"query": query},
                timeout=5,
            )
            if response.status_code >= 400:
                continue
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                continue

            best = max(
                payload,
                key=lambda item: _score_candidate(company_name, item.get("name", "")),
            )
            logo = best.get("logo")
            if logo:
                cache[key] = logo
                return logo
    except Exception:
        pass

    fallback_logo = (
        "https://ui-avatars.com/api/?background=E2E8F0&color=334155&size=128&name="
        + requests.utils.quote(company_name)
    )
    cache[key] = fallback_logo
    return fallback_logo


def _domain_from_url(url):
    try:
        parsed = urlparse(url or "")
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _resolve_company_logo_from_apply_url(apply_url, company_name, resolve_cache):
    cache_key = (apply_url or "").strip()
    if cache_key in resolve_cache:
        return resolve_cache[cache_key]

    domain = ""
    try:
        # Adzuna links often redirect to the publisher/company site.
        response = requests.get(apply_url, allow_redirects=True, timeout=8)
        domain = _domain_from_url(response.url)
    except Exception:
        domain = ""

    logo_url = None
    if domain and domain not in ("adzuna.com", "www.adzuna.com"):
        candidate = f"https://logo.clearbit.com/{domain}"
        try:
            check = requests.get(candidate, timeout=5)
            if check.status_code < 400:
                logo_url = candidate
        except Exception:
            logo_url = None

    resolve_cache[cache_key] = logo_url
    return logo_url

@app.post("/jobs/recommendations")
def job_recommendations():
    try:
        if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
            return jsonify(
                {
                    "error": (
                        "Adzuna credentials are missing. Set ADZUNA_APP_ID and "
                        "ADZUNA_APP_KEY."
                    )
                }
            ), 500

        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        location = (body.get("location") or "").strip()
        target_roles = body.get("targetRoles") or []
        role_keywords = body.get("roleKeywords") or []
        page = int(body.get("page") or 1)
        results_per_page = int(body.get("resultsPerPage") or 8)
        results_per_page = max(5, min(10, results_per_page))
        if page < 1:
            page = 1
        print(
            f"[jobs] incoming query='{query}' location='{location}' "
            f"page={page} resultsPerPage={results_per_page}"
        )

        base_params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        if location:
            base_params["where"] = location

        collected_items = []
        seen_ids = set()
        last_error = None
        for attempt_query in _build_adzuna_queries(query, target_roles, role_keywords):
            params = {**base_params, "what": attempt_query}
            response = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/{page}",
                params=params,
                timeout=20,
            )
            print(
                f"[jobs] adzuna status={response.status_code} "
                f"attempt_query='{attempt_query}'"
            )

            if response.status_code >= 400:
                last_error = {
                    "status": response.status_code,
                    "details": response.text[:400],
                }
                continue

            payload = response.json()
            attempt_items = payload.get("results", []) if isinstance(payload, dict) else []
            print(f"[jobs] raw results count={len(attempt_items)}")
            if attempt_items:
                for item in attempt_items:
                    item_id = str(item.get("id") or item.get("adref") or "")
                    if item_id and item_id in seen_ids:
                        continue
                    if item_id:
                        seen_ids.add(item_id)
                    collected_items.append(item)
                if len(collected_items) >= (results_per_page * 2):
                    break

        if not collected_items and last_error is not None:
            return jsonify(
                {
                    "error": f"Adzuna request failed ({last_error['status']})",
                    "details": last_error["details"],
                }
            ), 502

        normalized = []
        for item in collected_items:
            job = _normalize_adzuna_item(item)
            if job:
                normalized.append(job)

        normalized.sort(
            key=lambda job: _score_job(job, target_roles, role_keywords),
            reverse=True,
        )
        normalized = normalized[:results_per_page]

        logo_cache = {}
        redirect_logo_cache = {}
        for job in normalized:
            job["logoUrl"] = _resolve_company_logo_from_apply_url(
                job.get("applyUrl"),
                job.get("company"),
                redirect_logo_cache,
            ) or _lookup_logo_for_company(job.get("company"), logo_cache)

        print(f"[jobs] normalized count={len(normalized)}")
        if normalized:
            sample = normalized[0]
            print(
                "[jobs] sample "
                f"title='{sample.get('title')}' "
                f"company='{sample.get('company')}' "
                f"applyUrl='{sample.get('applyUrl')}'"
            )
        else:
            print("[jobs] normalized list is empty")

        return jsonify({"recommendations": normalized})
    except ValueError:
        return jsonify({"error": "Invalid numeric input for page/resultsPerPage"}), 400
    except Exception as e:
        print("JOBS ERROR:", e)
        return jsonify({"error": str(e)}), 500

# run
if __name__ == "__main__":
    app.run()