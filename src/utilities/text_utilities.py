import re
import humps


def get_nlg_service_name(service_name: str) -> str:
    domain, num = service_name.split("_")
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
