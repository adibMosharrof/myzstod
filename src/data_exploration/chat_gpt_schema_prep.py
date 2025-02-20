import os
import sys

from dotmap import DotMap

sys.path.insert(0, os.path.abspath("./src"))
import openai


class ChatSchemaPrep:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_prompt(self):
        return """
        The url for the schema file is: {self.cfg.schema_path}.
        The file contains schemas for different domains. 
        Give me 5 variations of schemas for each domain.
        When you create a new schema, keep the same number of slots and intents.
        For each slot and intent, use a different name but keep the same meaning.
"""

    def run(self):
        client = openai.OpenAI(
            api_key="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"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": self.get_prompt()}],
        )
        a = 1


if __name__ == "__main__":
    schema_path = "https://gist.githubusercontent.com/adibMosharrof/85a4f714060254759b14ce11ab1ae77f/raw/c84be750c627cf37c337c2f760ab1d48ef9d98df/bitod_schema"
    cgi = ChatSchemaPrep(DotMap(schema_path=schema_path))
    cgi.run()
