import hashlib
import os
import re
import humps
from torch import Tensor


def get_nlg_service_name(service_name: str) -> str:
    if "_" in service_name:
        domain, num = service_name.split("_")
    else:
        domain = service_name
    decamelized_domain = humps.decamelize(domain)
    spaced_domain = remove_underscore(decamelized_domain)
    return spaced_domain


def remove_underscore(item: str):
    return item.replace("_", " ")


def get_apicall_method_from_text(text: str, reg_exp=r"method='([^']+)'") -> str:
    try:
        match = re.search(reg_exp, text).group(1)
    except:
        try:
            reg_exp = r"method=([^,]+)"
            match = re.search(reg_exp, text).group(1)
        except:
            match = ""
        # match = ""
    return match


def get_parameters_from_text(text: str):
    reg_exp = r"(\w+)': '([^']+)'"
    try:
        matches = re.findall(reg_exp, text)
        out = dict(matches)
    except:
        out = {}
    return out


def remove_pad(tokens: list[Tensor], pad_token_id: int):
    return [row[row != pad_token_id] for row in tokens]


def hash_file_name(file_name: str, max_length: int, hash_length: int):
    if len(file_name) <= max_length:
        return file_name

    # Split the file name into base name and extension
    base_name, ext = os.path.splitext(file_name)

    # Create a hash of the base name
    hash_object = hashlib.md5(base_name.encode())  # You can use SHA256 if preferred
    hash_name = hash_object.hexdigest()[:hash_length]  # Truncate to 10 characters

    # Ensure the final name length is within the limit
    shortened_name = hash_name + ext
    if len(shortened_name) > max_length:
        raise ValueError(
            "Cannot shorten the file name to fit within the maximum length."
        )

    return shortened_name
