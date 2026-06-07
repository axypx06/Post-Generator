"""Few-shot post selector: loads processed posts and filters by length, language, and tag."""

import json
import pandas as pd


class FewShotPosts:
    """Loads processed LinkedIn posts and provides filtered few-shot examples."""

    def __init__(self, file_path="data/processed_posts.json"):
        self.df = None
        self.unique_tags = None
        self.load_posts(file_path)

    def load_posts(self, file_path):
        """Load posts from JSON, sanitize text encoding, and build the DataFrame."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw_posts = json.load(f)

        # .encode('ignore') safely drops corrupted emoji fragments like \ud83e
        for post in raw_posts:
            if "text" in post:
                post["text"] = post["text"].encode("utf-8", "ignore").decode("utf-8")

        self.df = pd.json_normalize(raw_posts)
        self.df['length'] = self.df['line_count'].apply(self.categorize_length)

        # Collect unique tags across all posts
        self.unique_tags = list(set(self.df['tags'].sum()))

    def get_filtered_posts(self, length, language, tag):
        """Return posts matching the given length category, language, and tag."""
        df_filtered = self.df[
            (self.df['tags'].apply(lambda tags: tag in tags)) &
            (self.df['language'] == language) &
            (self.df['length'] == length)
        ]
        return df_filtered.to_dict(orient='records')

    def categorize_length(self, line_count):
        """Map a line count to a length category: Short, Medium, or Long."""
        if line_count < 5:
            return "Short"
        if line_count <= 10:
            return "Medium"
        return "Long"

    def get_tags(self):
        """Return the list of all unique tags found across loaded posts."""
        return self.unique_tags


if __name__ == "__main__":
    fs = FewShotPosts()
    posts = fs.get_filtered_posts("Medium", "Hinglish", "Job Search")
    print(posts)
