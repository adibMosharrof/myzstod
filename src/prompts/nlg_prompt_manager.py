from prompts.prompt_constants import NlgPromptType


class NlgPrompt:
    def get_prompt(self, domain: str, schema: str, dialog_history: str) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}",
                schema,
                f"You have been provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                "Dialog History:",
                dialog_history,
                f"You have been provided with a Schema for domain: {domain}, an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "If there are search results, you should use information from the search results to generate the response.",
                "If you think you need more information to answer the user request, you can request information from the user.",
                "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request can only use the required slots and optional slots from that Intent.\n",
                "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
            ]
        )
        return prompt_text


class NlgMultidomainPrompt:
    def get_prompt(
        self,
        current_domain: str,
        other_domain: str,
        other_domain_schema: str,
        dialog_history: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain and other domains
        """

        prompt_text = "\n".join(
            [
                f"Previously, you were an expert chat assistant for the domain: {other_domain}",
                f"Here is the schema for the domain: {other_domain} is",
                f"{other_domain_schema}\n",
                f"Now you have switched domains and are an expert chat assistant for the domain: {current_domain}",
                f"The schema for the domain: {current_domain} is:",
                "The domain schemas contains intents and the required and optional slots for each intent.",
                "Instructions: You will be given an incomplete dialog between a user and a chat assistant, and sometimes you will also be given search results.",
                "As an expert, you must generate the most appropriate response for the chat assistant.",
                "If there are search results, you must use information from the search results to generate the response.",
                "If you think you need more information to answer the user request, you can request information from the user. However, your request can only use the required and optional slots from the schema.",
                "Dialog History:",
                dialog_history,
                "End of Dialog History",
                "Please generate the response for the chat assistant.",
            ]
        )
        return prompt_text


class NlgPromptFactory:
    @classmethod
    def get_handler(self, prompt_type: str) -> NlgPrompt:
        if prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            prompt_cls = NlgMultidomainPrompt()
        elif prompt_type == NlgPromptType.DEFAULT.value:
            prompt_cls = NlgPrompt()
        else:
            all_prompts = ",".join(NlgPromptType.list())
            raise NotImplementedError(
                f"Prompt type {prompt_type} is not implemented. It must be one of the following values {all_prompts}."
            )
        return prompt_cls
