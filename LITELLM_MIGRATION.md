# LiteLLM Migration Notes

## 1. Add dependency

Add LiteLLM first so the backend can import it before any code is switched over. This is just the runtime dependency that makes the new calls available.

**File**: [backend/requirements.txt](backend/requirements.txt#L1)

**Before**
```txt
google-generativeai>=0.3.0
google-genai
```

**After**
```txt
google-generativeai>=0.3.0
google-genai
litellm
```

## 2. Change code

Swap the Gemini SDK calls for LiteLLM calls. The main idea is the same in all four files: keep the prompt logic, but change how the model is called and how the response is read back. The extra helper functions are there to normalize model names, pick the right API key, and extract text from LiteLLM responses.

### 2.1 `backend/gemini_summarizer.py`

Why: this file is the simplest text-only path, so it is the easiest place to prove the LiteLLM workflow works.

**Before**
```python
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)
response = model.generate_content(prompt)
summary = response.text.strip()
```

**After**
```python
import litellm

model_name = os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"))

response = litellm.completion(
	model=model_name,
	messages=[{"role": "user", "content": prompt}],
)
summary = response.choices[0].message.content.strip()
```

### 2.2 `tools/scene_description.py`

Why: this is the main image-description path, so it shows how vision requests move from Gemini image objects to LiteLLM vision payloads.

**Before**
```python
import google.generativeai as genai

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)
response = model.generate_content([prompt, pil_image])
description = response.text.strip()
```

**After**
```python
import base64
import litellm

def _image_to_data_uri(image: np.ndarray) -> str:
	success, buffer = cv2.imencode(".jpg", image)
	image_base64 = base64.b64encode(buffer).decode("ascii")
	return f"data:image/jpeg;base64,{image_base64}"

response = litellm.completion(
	model=model_name,
	messages=[
		{
			"role": "user",
			"content": [
				{"type": "text", "text": prompt},
				{"type": "image_url", "image_url": {"url": _image_to_data_uri(processed_image)}},
			],
		}
	],
)
description = response.choices[0].message.content.strip()
```

### 2.3 `tools/clothing_recognition.py`

Why: this should follow the same vision path as scene description so both tools behave consistently.

**Before**
```python
import google.generativeai as genai

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)
response = model.generate_content([prompt, pil_image])
description = response.text.strip()
```

**After**
```python
import base64
import litellm

def _image_to_data_uri(image: np.ndarray) -> str:
	success, buffer = cv2.imencode(".jpg", image)
	image_base64 = base64.b64encode(buffer).decode("ascii")
	return f"data:image/jpeg;base64,{image_base64}"

response = litellm.completion(
	model=model_name,
	messages=[
		{
			"role": "user",
			"content": [
				{"type": "text", "text": prompt},
				{"type": "image_url", "image_url": {"url": _image_to_data_uri(processed_image)}},
			],
		}
	],
)
description = response.choices[0].message.content.strip()
```

### 2.4 `backend/stream_server.py`

Why: this is the server entry point, so it is where model selection and the remaining Gemini-backed parsing calls need to be centralized.

**Before**
```python
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

if GEMINI_API_KEY:
	genai.configure(api_key=GEMINI_API_KEY)

gemini_live_manager = GeminiLiveManager(GEMINI_API_KEY) if GEMINI_API_KEY else None
```

**After**
```python
import litellm

LLM_MODEL = os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"))

# provider keys stay in .env, model changes through LLM_MODEL
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
```

**Before**
```python
response = model.generate_content(prompt)
summary = response.text.strip()
```

**After**
```python
response = litellm.completion(
	model=LLM_MODEL,
	messages=[{"role": "user", "content": prompt}],
)
summary = response.choices[0].message.content.strip()
```

## 3. Update env vars

Keep all provider keys in `.env`, then switch the active model by changing only `LLM_MODEL`. That is the part that makes the migration flexible and lets you test different providers without touching code.

**Files**:

- [README.md](README.md#L71)
- [backend/README.md](backend/README.md#L20)

**Before**
```bash
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-3-flash-preview
```

**After**
```bash
LLM_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

**Optional default in code**
```python
model_name = os.environ.get("LLM_MODEL", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"))
```

## Extra helpers used in code

These are small implementation details that support the three steps above:

- `_resolve_model_name(...)`: keeps Gemini model names working with LiteLLM provider prefixes.
- `_resolve_api_key(...)`: chooses the matching provider key from `.env`.
- `_extract_text(...)`: reads the final text back from LiteLLM responses.
- `_image_to_data_uri(...)`: turns a local image into a LiteLLM vision payload.
- `_pil_image_to_data_uri(...)`: same idea for the follow-up image question in `stream_server.py`.

Implementation note:

- Backend helper logic is now centralized in [backend/litellm_utils.py](backend/litellm_utils.py#L1), and imported by [backend/gemini_summarizer.py](backend/gemini_summarizer.py#L1) and [backend/stream_server.py](backend/stream_server.py#L1).