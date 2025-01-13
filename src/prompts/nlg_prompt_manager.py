from prompts.auto_tod_prompt import AutoTodPrompt
from prompts.chatgptv2_prompt import ChatGptV2Prompt
from prompts.prompt_constants import NlgPromptType
from my_enums import ContextType
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)
from utilities.context_manager import ContextManager


class CrossAttentionPrompt:
    def get_generation_prompt(
        self,
        domain: str,
        dialog_history: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                "Dialog History:",
                dialog_history,
            ]
        )
        return prompt_text

    def get_schema_prompt(
        self,
        domain: str,
        schema: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                schema,
            ]
        )
        return prompt_text


class PseudoLabelsPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat bot for the domain: {domain}.",
                # "Instructions: You must generate the most appropriate response for the chat bot.",
                # "When making API calls, use the intent pseudo_name and slot pseudo name."
                f"Here is the schema for {domain}:.",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat bot, and an optional search results.",
                "Dialog History:",
                dialog_history,
                ". Instructions: Using the Dialog History, Search Results, please generate the response for the chat bot.",
                "The response can be an api call or a response to the user.",
            ]
        )
        return prompt_text


class NlgPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}.",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call or a response to the user.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request should use the required slots and optional slots from that Intent.",
                # f"You will be provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                f"You will be provided with a Schema for domain: {domain}.",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Dialog History:",
                dialog_history,
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                ". Using the Dialog History, Search Results, and by following the Instructions please generate the response for the chat assistant.",
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
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
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


class ChatGptPrompt:
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
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = (
            f"You are an expert chat assistant for the DOMAIN: {domain}. \n"
            f"Instructions:\nGenerate a natural and helpful SYSTEM response for the given task-oriented dialog context.\n"
            f"Here are important slots related to each intent:\n"
        )
        domains = domains_original.split(",")
        for dom in domains:
            dstc_schema = all_schema[dom]
            # for intent, req_slots, opt_slots in intents_slots:
            for dstc_intent in dstc_schema.intents:
                req_slots_str = ", ".join(dstc_intent.required_slots)
                opt_slots_str = ", ".join(dstc_intent.optional_slots.keys())
                prompt_text += (
                    f"\nIntent: {dstc_intent.name}\n"
                    f"required slots: {req_slots_str}\n"
                    f"optional slots: {opt_slots_str}\n"
                )

        prompt_text += (
            f"\nYou can request the values of any number of slots from the USER in order to fulfill the user's current intent."
            f"Generally, required slots are more important than the optional ones. Moreover, you should restrict your conversation to the slots that are relevant to the user's current intent. "
        )
        prompt_text += (
            f"\nYou can assume that you have access to the following API calls: "
        )

        # for intent, _, _ in intents_slots:
        for dom in domains:
            dstc_schema = all_schema[dom]
            # for intent, req_slots, opt_slots in intents_slots:
            for dstc_intent in dstc_schema.intents:
                req_slots_str = ", ".join(dstc_intent.required_slots)
                opt_slots_str = ", ".join(dstc_intent.optional_slots.keys())
                prompt_text += f"method={dstc_intent.name}, parameters={req_slots_str}, {opt_slots_str}. "

        prompt_text += (
            f"Use column names as parameters for API calls. While making the call, Make sure you ask all the required slots from the user before making an API call, you can skip unwanted parameters. "
            f"Here is an example: ApiCall(method='intent_i', parameters={{'slot_1': 'value_1', 'slot_2': 'value_2', ...}}). "
            f"Try to match the parameters in the API calls with the column names from the dataframe. Match the required and optional slots with the column names and use the columns names to search in the API calls.Use date format YYYY-MM-DD if needed to search in API call. "
            f"When you think you have enough information about slot values, you must output the API call. Right after your API call, you will be provided with search result of the API call. "
            f"You will be provided with an incomplete dialog context between USER and SYSTEM, and optionally, search results from dataframe. "
            f"Here a few general guidelines: Don't overload the USER with too many choices. Confirm the values of slots and Ask the USER to confirm their choice before finalizing the transaction. "
            f"Whenever confused, it is always better to ask or confirm with the user. If you feel like that user is confused, you may guide the user with relevant suggestions. "
            f"Task:\n Based on the last USER utterance and dialog context you have to generate a conversational response or an API call in order to fulfill the user's current intent. "
            f"You may need to make API calls and use the results of the API call. \n Dialog Context: {dialog_history}."
        )

        return prompt_text


class BaselineTodPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str = "",
        dialog_history: str = "",
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        schema_prompt = " and the schemas," if schema else ""
        prompt_text = "\n".join(
            [
                schema or "",
                f"Instructions: Given the dialog history{schema_prompt} please generate the system response. The response can be an api call or a response to the user.\n\n",
                "Begin Context",
                "Dialog History",
                dialog_history,
            ]
        )
        return prompt_text


class NlgPromptFactory:
    @classmethod
    def get_handler(
        self, prompt_type: str, context_type: ContextType = ContextType.NLG_API_CALL
    ) -> NlgPrompt:
        if prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            return NlgMultidomainPrompt()

        if prompt_type == NlgPromptType.CHATGPT.value:
            return ChatGptPrompt()
        if prompt_type == NlgPromptType.CHATGPTV2.value:
            return ChatGptV2Prompt()
        if prompt_type == NlgPromptType.CROSS.value:
            return CrossAttentionPrompt()
        if prompt_type == NlgPromptType.AUTO_TOD.value:
            return AutoTodPrompt()

        if context_type in [
            ContextType.KETOD_API_CALL.value,
            ContextType.KETOD_GPT_API_CALL.value,
        ]:
            return KetodPrompt()
        if ContextManager.is_sgd_pseudo_labels(context_type):
            return PseudoLabelsPrompt()
        if any(
            [
                ContextManager.is_zstod(context_type),
                ContextManager.is_simple_tod(context_type),
                ContextManager.is_soloist(context_type),
            ]
        ):
            return BaselineTodPrompt()
        if prompt_type == NlgPromptType.DEFAULT.value:
            return NlgPrompt()
        all_prompts = ",".join(NlgPromptType.list())
        raise NotImplementedError(
            f"Prompt type {prompt_type} is not implemented. It must be one of the following values {all_prompts}."
        )
