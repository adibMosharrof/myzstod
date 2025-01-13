from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)


class ChatGptV2Prompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
        all_data=None,
    ) -> str:

        return ""
