from dotmap import DotMap
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)

from utilities import text_utilities

TEMPLATE = """You are an intelligent AI Assistant to help the user complete complex tasks. There are many services to fulfill the user's goals. Each service consists of mulitple functions that the AI Assistant can call. The AI Assistant can choose to call a function in order to provide information or make a transaction for the user.

The functions are divided into query function and transaction function. The query function will return the records in the database that meets the conditions, and the transaction function will return the corresponding reference number if calling successfully.

Today is 2019-03-01, Friday. (This Saturady is 2019-03-02. This Sunday is 2019-03-03.)
When specifying an data type parameter without given full date, prefix as "2019-03-xx".

{services_info}

# Remember

- Don't make assumptions about what values to plug into functions. Ask for clarification if any parameter is missing or ambiguous.
- Before calling a transaction function for the user, such as transfer money and book restaurants, the AI Assistant MUST show all the function parameters and confirm with the user.
- You must not call the same function with the same parameters again and again.
- When finishing the user's goals, saying goodby to the user and finish the dialogue.
"""

SERVICE_TEMPLATE = """# Service: {service_name}

## Description

{service_desc}

## Functions

{functions_info}
"""


EXAMPLE = """
# Service: Buses_1

## Description

Book bus journeys from the biggest bus network in the country.

## Functions

- FindBus: Find a bus journey for a given pair of cities. (Query function)
- BuyBusTicket: Buy tickets for a bus journey. (Transaction function)
"""


class AutoTodPrompt:
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
        service_names = domains_original.split(",")
        out = DotMap()
        out.system_prompt = self.system_prompt(service_names, all_schema)
        out.function_prompt = self.make_function_schemas(service_names, all_schema)
        out.dialog_history = dialog_history
        return out

    def system_prompt(
        self, service_names: list[str], all_schema: dict[str, DstcSchema]
    ) -> str:

        schemas = {schema_str: all_schema[schema_str] for schema_str in service_names}
        services_info = []
        for service_name in service_names:
            service_schema = schemas[service_name]
            functions_info = []
            short_service_name = text_utilities.get_nlg_service_name(service_name)
            for intent in service_schema.intents:
                func_info = f"- {intent.name}: {intent.description}"

                if not intent.is_transactional:
                    func_info += " (Query function)"
                else:
                    func_info += " (Transaction function)"
                functions_info.append(func_info)
            functions_info = "\n".join(functions_info)
            service_info = SERVICE_TEMPLATE.format(
                service_name=short_service_name,
                service_desc=service_schema.description,
                functions_info=functions_info,
            )
            services_info.append(service_info)
        services_info = "\n\n".join(services_info)

        prompt = TEMPLATE.format(services_info=services_info)
        return prompt

    def make_function_schemas(
        self, service_names: list[str], all_schema: dict[str, DstcSchema]
    ) -> str:
        functions = []

        for service_name in service_names:
            service_schema = all_schema[service_name]
            for intent in service_schema.intents:
                func_schema = self.make_one_function_schema(service_schema, intent.name)
                functions.append(func_schema)

        return functions

    def make_one_function_schema(self, service_schema, intent_name):
        intent_dict = {intent.name: intent for intent in service_schema.intents}
        intent = intent_dict[intent_name]

        func_schema = {
            "name": f"{intent_name}",
            "description": None,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": intent.required_slots.copy(),
            },
        }

        desc = intent.description + "."
        if not intent.is_transactional:
            desc += " (Query function. Return db recored that meets conditions.)"
        else:
            desc += " (Transaction function. Return a reference number when calling succeeds.)"
        func_schema["description"] = desc

        slot_dict = {slot.name: slot for slot in service_schema.slots}
        for slot_name in intent.required_slots + list(intent.optional_slots.keys()):
            slot = slot_dict[slot_name]

            # Schema
            property_schema = {
                "description": slot.description,
            }

            func_schema["parameters"]["properties"][slot_name] = property_schema

        return func_schema
