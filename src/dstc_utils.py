def get_dstc_service_name(service_name: str) -> str:
    return service_name[: service_name.find("_")]
