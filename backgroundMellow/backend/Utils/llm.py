import os
import json
import logging
import dotenv
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

def query_llm(
    llm_name: str,
    model_name: str,
    prompt: str,
    api_key: str = "",
    system_prompt: str = "",
):
    """
    Query a supported LLM (gemini, gpt, grok, etc.) and return parsed JSON response.
    llm_name: one of "gemini", "gpt", "grok"
    model_name: model version/name for the target llm
    prompt: user input prompt
    api_key: (optional) API key for selected LLM, overrides env
    system_prompt: (optional) system prompt if supported (e.g. for chatgpt)
    Returns:
        dict (parsed JSON object) or None on error.
    """
    llm_name = llm_name.lower()
    result = None

    if llm_name == "gemini":
        try:
            # Import locally to avoid issues if package not installed
            import google.genai as genai
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                logger.error("Gemini API key not found in environment.")
                return None
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model=model_name, contents=prompt)
            response_text = getattr(response, "text", None)
            if not response_text:
                logger.error("No response text from Gemini.")
                return None
            # Many LLMs wrap json in markdown; strip it if needed
            json_str = response_text.replace("```json", "").replace("```", "").strip()
            # Try to extract array/object from within possible extra text
            import re
            match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", json_str)
            if match:
                json_str = match.group()
            result = json.loads(json_str)
        except Exception as e:
            logger.error(f"Gemini query failed: {e}", exc_info=True)
            return None

    elif llm_name == "gpt":
        try:
            # Import locally to avoid issues if package not installed
            import openai
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                logger.error("OpenAI API key not found in environment.")
                return None
            openai.api_key = key
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
            )
            response_text = completion.choices[0].message.content
            if response_text:
                json_str = response_text.replace("```json", "").replace("```", "").strip()
            else:
                logger.error("No response text from ChatGPT.")
                return None
            import re
            match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", json_str)
            if match:
                json_str = match.group()
            result = json.loads(json_str)
        except Exception as e:
            logger.error(f"ChatGPT query failed: {e}", exc_info=True)
            return None

    else:
        logger.error(f"Unsupported LLM name: {llm_name}")
        return None

    return result