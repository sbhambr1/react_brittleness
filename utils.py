import tiktoken 

def count_tokens(string: str, encoding_name: str = None) -> int:
    encoding_name = 'cl100k_base' if  encoding_name is None else encoding_name
    
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens