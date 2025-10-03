import tiktoken

def main():
    encoding = tiktoken.get_encoding("cl100k_base")

    text_example = "Hello world! Tokenization is great."

    print(f"Original text: '{text_example}'")
    print("-" * 30)

    token_ids = encoding.encode(text_example)

    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    print("-" * 30)

    decoded_text = encoding.decode(token_ids)

    print(f"Decoded text: '{decoded_text}'")
    print("-" * 30)

    print("Individual tokens:")
    for token_id in token_ids:
        token_str = encoding.decode([token_id]) # Decode each ID individually
        print(f"  ID: {token_id}, token: '{token_str}'")

if __name__ == "__main__":
    main()