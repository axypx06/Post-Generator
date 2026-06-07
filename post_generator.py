"""LinkedIn post generator using few-shot examples and an LLM."""

from call_llm import llm
from few_shot import FewShotPosts

few_shot = FewShotPosts()

_LENGTH_RANGES = {
    "Short": "1 to 5 lines",
    "Medium": "6 to 10 lines",
    "Long": "11 to 15 lines",
}


def get_length_str(length):
    """Return a human-readable line-range string for the given length category."""
    return _LENGTH_RANGES.get(length, "6 to 10 lines")


def generate_post(length, language, tag):
    """Invoke the LLM with a fully built prompt and return the generated post text."""
    prompt = get_prompt(length, language, tag)
    response = llm.invoke(prompt)
    return response.content


def get_prompt(length, language, tag):
    """Build the full generation prompt with instructions and up to 2 few-shot examples."""
    length_str = get_length_str(length)

    language_instruction = (
        "The post must be a natural mix of Hindi and English (Hinglish). "
        "Use the English script only — do not use Devanagari."
        if language == "Hinglish"
        else "The post must be written in English."
    )

    prompt = f"""You are an expert LinkedIn content creator who writes authentic, engaging posts.

Generate a LinkedIn post with the following requirements:
1. Topic: {tag}
2. Length: {length_str}
3. Language: {language_instruction}
4. Tone: Conversational and relatable — avoid corporate jargon or motivational fluff.
5. Format rules:
   - Open with a strong hook that stops the scroll.
   - Use short paragraphs or single-line breaks for easy reading.
   - Close with a reflection, question, or subtle call-to-action.
   - Add 2–3 relevant hashtags on the last line.
   - Use emojis sparingly and only where they feel natural.
6. Do NOT start the post with "I".
7. Output the post text only. No preamble, no explanation, no label.
"""

    examples = few_shot.get_filtered_posts(length, language, tag)

    if examples:
        prompt += "\n8. Mirror the writing style shown in the examples below:\n"
        for i, post in enumerate(examples[:2]):  # cap at 2 examples
            prompt += f"\n--- Example {i + 1} ---\n{post['text']}\n"

    return prompt


if __name__ == "__main__":
    print(generate_post("Medium", "English", "Mental Health"))
