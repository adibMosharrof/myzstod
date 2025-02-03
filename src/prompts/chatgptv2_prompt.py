from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)
import pandas as pd
import random


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
        """
        Returns the NLG prompt for the given domain
        """
        
        ExampleDialogue = ""

        #if domain in all_data.keys():
            #apicallexamples = all_data[domain]
        #else:
            #apicallexamples = all_data['default']
        # Split the domain if it contains multiple values
        
        uniq_domain = domain.split(",")

        # Collect examples from the dictionary for each matched domain
        matched_examples = [all_data[d] for d in uniq_domain if d in all_data]

        # If examples are found, join them; otherwise, use the default value
        apicallexamples = "\n\n".join(matched_examples) if matched_examples else all_data['default']
        


        prompt_text = "\n\n".join(
            [
                # Need to call prompt from python--
                f"You are an expert chat assistant for the domain: {domain}",
                "Instructions:\nAs an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call or a response to the user.",
                "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request should use the required slots and optional slots from that Intent.",
                # Basic Info Schema, Slots and Intent
                f"You will be provided with a Schema for domain: {domain}",
            ]
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
                    f"is transactional: {dstc_intent.is_transactional}\n"
                    f"required slots: {req_slots_str}\n"
                    f"optional slots: {opt_slots_str}\n"
                )

        prompt_text += "\n\n".join(
            [
                "General Guidenlines-\n",
                "\n",
                "Go through each of them. Look extra carefully for the examples provided within the guidelines and also as a separate section in the bottom. Try to match the gold responses as much as possible,\n",
                "\n",
                "Use column names as parameters for API calls. While making the call, Make sure you ask all the required slots from the user before making an API call, you can skip unwanted parameters. ,\n",
                "            \n",
                "Here is an example: ApiCall (method='intent_i', parameters={{ 'slot_1': 'value_1', 'slot_2': 'value_2', ... }}).\",\n",
                "\n",
                'Try to match the parameters in the API calls with the column names from the dataset. Match the required and optional slots with the column names and use the columns names to search in the API calls. Use date format YYYY-MM-DD if needed to search in API call",\n',
                "\n",
                "While asking conversational questions add some general suggestions for the USER. Such as, instead of just saying 'What kind of restaurants are you looking for?' add 'What kind of cuisine would you prefer? Chinese? Malaysian?'\",\n",
                "\n",
                "Filling up all required slots before making the api call is a must. For example, look at this generalized schema and conversation\\nHere the intent is X and the required slots are C and A. It is not a transactional step. So you have to ask for A and C to fillup the slots. When the USER answers about them and you know for sure only then you make the API call.\\n Schema:\\nIntent: X\\nrequired slots: A, C\\nis transactional: 0\\n\\nGold Response: Which 'A' are you looking in?\\nUser: 'B', please.\\nGold Response: What kind of 'C' would you like? 'D', 'E', or something else?\\nUser: 'F' sounds good right now.\\nGold Response: ApiCall(method='X', parameters={'A': 'B', 'C': 'F'})\\n\\nHere, instead of actual information placeholders like 'A', 'B', 'C', 'D', 'E' and 'F' has been used\\n\\nthe api call has only been made after 'A' and 'C' has been fixed. Both 'A' and 'C' are required slots for the intent 'X'.\",\n",
                "\n",
                'For intents that are not transactional i.e have is transactional as 0(False), if you have all the required slots filled up; make the api call promptly and do not ask the user again for reconfirmation",\n',
                "\n",
                'For intents that have is transactional as 1(True), follow the following instructions_",\n',
                "\n",
                'Be extra cautious about making the API call. Untill the USER explicitly approves with words such as Yes/Alright/Go Ahead etc, wait till then to make the API call.",\n',
                "\n",
                'Always reconfirm with the USER about their choices before the final step.",\n',
                "\n",
                "If the user changes something of the required slots, reconfirm the whole details again from scratch before making the API call. \n",
                "\n",
                "If you are using date for the API call always use \n",
                "\n",
                "If the user requirements change or differ from a previous API call or a previous search from an API call does not return required outcome, feel free to make new API calls with the changed slot values\n",
                "\n",
                "If an API call search does not return what the user asks for, make new api call\n",
                "Here are some example API calls for you. There will be some conversations and afterwards an example of the required API call, so that you can learn where and how to make the API calls"
                "\n",
                "Within the api call parameters, always use single quotation marks only and never double ones\n",
                "\n",
                "Do not consider the present date. Consider the month to be March 2019",
                apicallexamples,
                "\n",
                'So the principle to keep in mind is\\n 1_provide option. 2_if users agree then present all the details and then ask if they want to go ahead with it. (Basically Reconfrim)3_Only after that confirmation make the API call.",\n',
                "\n",
                "But if the user confirms all the details once, then do not wait for the optional slots. Make the API call directly. For example\\nSchema: Z\\nIntent: X\\nrequired slots: A,B,C\\noptional slots: D,E\\nis transactional: 1\\nIntent:Y\\nrequired slots: ...\\noptional slots: K\\nis transactional:0\\n\\nUser: That suits me well. Can you tell me if they have 'K'?\\nGold Response: ApiCall(method='X', parameters={'B': 'F', 'D': 'G', 'E': 'H', 'A': 'I', 'C': 'J'})\\nSearch Results\\n[{... 'K': 'False', ...}]\\nEnd Search Results\\nGold Response: Your transaction has been made without errors, but unfortunately they do not have 'K'.\\n\\nHere although an optional slot is false, but as the USER has already confirmed the details about the transaction, the API call has been shoot off.\",\n",
                "\n",
                'Right after your API call, you will be provided with search result of the API call. You will be provided with an incomplete dialog context between USER and SYSTEM, and optionally, search results from dataframe",\n',
                "\n",
                'When offering USER with options from the search result offer the top result only. And only one result at once. Offer the other ones one by one if the USER asks. So the principle to keep in mind is first, provide option. Secondly, if users agree then present all the details and then ask if they want to go ahead with it. Only after that confirmation make the API call.",\n',
                "\n",
                'Do not suggest an option more than once if it has already been suggested",\n',
                "\n",
                'While providing options from a search result provide the option only and do not provide details. i.e. Keep it as short as possible.",\n',
                "\n",
                "As a general guideline, keep your conversational responses as short and small as possible. Omit all optative sentences. While responding your target should be keeping the word count as low as possible\n",
                "\n",
                "Do not use your knowledge of your own only use the search result given to you as part of the dialog history. For example, don't actually make api calls and try to search real time\",\n",
                "\n",
                'Read the full dialog history before starting to generate a response",\n',
                "\n",
                "At the end of the conversation do not forget to exchange greetings with the USER\n",
                "\n",
                'That is all the examples and instructions for you to understand the gold responses, we will now move to the tasks\\n\\n"\n',
                "\n",
                "#Main Task\n",
                "\n",
                "\n",
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.\\nDialog History:{dialog_history}",
                "\n",
                "Using the Dialog History, Search Results, and by following the Instructions please generate the response for the last user utterance for the chat assistant.\n",
                "Do not use your own knowledge here and only use the history, search result as your source of information. Also you do not have to actually carry out an API call, Only pretend that you are making the API calls.\n",
            ]
        )

        return prompt_text
