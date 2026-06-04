import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    msgs = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=msgs,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "description": "Read and return the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path to the file to read",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                }
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        if chat.choices[0].finish_reason == "tool_calls":
            tool_calls = chat.choices[0].message.tool_calls
            for tc in tool_calls:
                if tc.function.name == "Read":
                    file_path = json.loads(tc.function.arguments)["file_path"]
                    if os.path.isfile(file_path):
                        with open(file_path) as f:
                            tc_resp = {
                                "role": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": f.read(),
                            }
                            msgs.append(tc_resp)
        elif chat.choices[0].finish_reason == "stop":
            print(chat.choices[0].message.content)
            return 0
        else:
            raise RuntimeError("unknown finish reason")

    # # You can use print statements as follows for debugging, they'll be visible when running tests.
    # print("Logs from your program will appear here!", file=sys.stderr)


if __name__ == "__main__":
    main()
