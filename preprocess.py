"""Preprocess raw LinkedIn posts by extracting metadata and unifying tags via LLM."""

import re
import json
import logging
from call_llm import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_parse_json(text):
    """Try multiple strategies to extract a JSON object from an LLM response string.

    Returns the parsed dict, or None if all strategies fail.
    """
    # LangChain JsonOutputParser — handles ```json ... ``` fences
    try:
        return JsonOutputParser().parse(text)
    except OutputParserException:
        pass

    # direct json.loads after stripping whitespace
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # regex — extract the first {...} block from the response
    # Handles cases where the LLM wraps JSON inside prose or code blocks
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def process_posts(raw_file_path, processed_file_path=None):
    """Load raw posts, enrich each with LLM-extracted metadata, unify tags, and save."""
    if not processed_file_path:
        raise ValueError("processed_file_path must be provided")

    with open(raw_file_path, encoding="utf-8") as f:
        posts = json.load(f)

    enriched_posts = []
    for i, post in enumerate(posts):
        try:
            logger.info("Extracting metadata for post %d/%d", i + 1, len(posts))
            metadata = extract_metadata(post["text"])
            enriched_posts.append({**post, **metadata})
        except (OutputParserException, KeyError, ValueError) as e:
            logger.error("Post %d metadata extraction failed: %s. Skipping enrichment.", i + 1, e)
            enriched_posts.append(post)

    unified_tags = get_unified_tags(enriched_posts)

    for post in enriched_posts:
        current_tags = post.get("tags", [])
        post["tags"] = list({unified_tags.get(tag, tag) for tag in current_tags})

    with open(processed_file_path, "w", encoding="utf-8") as outfile:
        json.dump(enriched_posts, outfile, indent=4, ensure_ascii=False)

    logger.info("Done. Saved %d posts to %s", len(enriched_posts), processed_file_path)


def extract_metadata(post_text):
    """Extract line_count, language, and tags from a single post string using the LLM."""
    template = '''Your task is to extract metadata from a LinkedIn post.

OUTPUT RULES — read carefully:
- Output ONLY a raw JSON object. Nothing else.
- Do NOT write code. Do NOT explain. Do NOT use markdown. Do NOT add backticks.
- Start your response with {{ and end with }}

Required JSON keys:
- "line_count": integer, number of lines in the post
- "language": exactly "English" or "Hinglish" (Hinglish = Hindi + English mixed)
- "tags": JSON array of at most 2 short topic strings

Correct example response:
{{"line_count": 5, "language": "English", "tags": ["Job Search", "Motivation"]}}

Post:
{post}'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post_text})

    res = _safe_parse_json(response.content)
    if res is None:
        raise OutputParserException(
            "Unable to parse metadata JSON from LLM response."
        )

    res.setdefault("line_count", post_text.count("\n") + 1)
    res.setdefault("language", "English")
    res.setdefault("tags", [])

    if not isinstance(res["tags"], list):
        res["tags"] = [res["tags"]] if isinstance(res["tags"], str) else []

    return res


def get_unified_tags(posts_with_metadata):
    """Collect all unique tags across posts and merge similar ones via the LLM."""
    unique_tags = set()
    for post in posts_with_metadata:
        tags = post.get("tags", [])
        if isinstance(tags, list):
            unique_tags.update(tags)

    if not unique_tags:
        logger.warning("No tags found across posts. Skipping unification.")
        return {}

    unique_tags_list = ", ".join(sorted(unique_tags))

    template = '''Your task is to unify a list of LinkedIn post tags.

OUTPUT RULES — read carefully:
- Output ONLY a raw JSON object. Nothing else.
- Do NOT write code. Do NOT explain. Do NOT use markdown. Do NOT add backticks.
- Start your response with {{ and end with }}

Merging rules:
- Merge semantically similar tags into one representative tag.
  "Jobseekers", "Job Hunting" -> "Job Search"
  "Motivation", "Inspiration", "Drive" -> "Motivation"
  "Personal Growth", "Personal Development", "Self Improvement" -> "Self Improvement"
  "Scam Alert", "Job Scam" -> "Scams"
  "AI", "GenAI", "LLM", "ChatGPT", "AI Agents" -> "Artificial Intelligence"
  "Prompt Engineering", "System Prompt" -> "Prompt Engineering"
- Every unified tag must be in Title Case.
- Every original tag must appear as a key in the output.

Correct example response:
{{"Jobseekers": "Job Search", "Job Hunting": "Job Search", "Motivation": "Motivation"}}

Tags to unify:
{tags}'''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": unique_tags_list})

    res = _safe_parse_json(response.content)

    if res is None:
        # Fallback: identity mapping — each tag maps to itself rather than crashing
        logger.warning(
            "Could not parse unified tags from LLM. Falling back to identity mapping."
        )
        return {tag: tag for tag in unique_tags}

    # Ensure every original tag has a mapping even if LLM dropped some
    for tag in unique_tags:
        res.setdefault(tag, tag)

    return res


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
