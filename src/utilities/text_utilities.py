import humps


def get_nlg_service_name(service_name: str) -> str:
    domain, num = service_name.split("_")
    decamelized_domain = humps.decamelize(domain)
    spaced_domain = remove_underscore(decamelized_domain)
    return spaced_domain


def remove_underscore(item: str):
    return item.replace("_", " ")
