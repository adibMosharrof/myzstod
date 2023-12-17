from prompts.prompt_constants import NlgPromptType
from my_enums import ContextType


class NlgPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call or a response to the user.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                # "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request can only use the required slots and optional slots from that Intent.",
                # f"You will be provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                f"You will be provided with a Schema for domain: {domain}",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Dialog History:",
                dialog_history,
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                "Using the Dialog History, Search Results and by following the Instructions please generate the response for the chat assistant.",
            ]
        )
        return prompt_text


class NlgMultidomainPrompt:
    def get_prompt(
        self,
        current_domain: str,
        current_domain_schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain and other domains
        """

        prompt_text = "\n".join(
            [
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be a service call or a response to the user.",
                f"Previously, you were an expert chat assistant for the domain: {other_domain}",
                other_domain_schema,
                f"You were provided with a Schema for the domain: {other_domain}.",
                f"Now you have switched domains and are an expert chat assistant for the domain: {current_domain}",
                current_domain_schema,
                f"You have been provided with a Schema for domain: {current_domain}, which is in the same format as the {other_domain} and contains the relevant Intents for the {current_domain}.",
                f"Use the {current_domain} schema in a similar way you used it in the {other_domain}.",
                "Dialog History:",
                dialog_history,
                f"You have been provided with a Schema for domain: {current_domain}, an incomplete dialog between a user and a chat assistant, and an optional search results.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                # "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request can only use the required slots and optional slots from that Intent.\n",
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                "Using the Dialog History, Search Results and by following the Instructions please generate the response for the chat assistant.",
            ]
        )

        return prompt_text


class KetodPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call, entity query or a response to the user.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request should use the required slots and optional slots from that Intent.",
                # f"You will be provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                f"You will be provided with a Schema for domain: {domain}",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Dialog History:",
                dialog_history,
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                "Using the Dialog History, Search Results and by following the Instructions please generate the response for the chat assistant.",
            ]
        )
        return prompt_text


class NlgPromptFactory:
    @classmethod
    def get_handler(
        self, prompt_type: str, context_type: ContextType = ContextType.NLG_API_CALL
    ) -> NlgPrompt:
        if context_type == ContextType.KETOD_API_CALL.value:
            return KetodPrompt()
        if prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            return NlgMultidomainPrompt()
        if prompt_type == NlgPromptType.DEFAULT.value:
            return NlgPrompt()
        all_prompts = ",".join(NlgPromptType.list())
        raise NotImplementedError(
            f"Prompt type {prompt_type} is not implemented. It must be one of the following values {all_prompts}."
        )
