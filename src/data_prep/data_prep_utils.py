from my_enums import SpecialTokens, Steps, ZsTodConstants

from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
    get_schemas,
)


def extract_from_target(target: str, start_tokens: list[str], end_tokens: list[str]):
    texts = []
    for start_token, end_token in zip(start_tokens, end_tokens):
        try:
            start_index = target.index(start_token)
            end_index = target.index(end_token)
            texts.append(target[start_index : end_index + len(end_token)]),
        except ValueError:
            texts.append("")
    return "".join(
        [
            SpecialTokens.begin_target,
            "".join(texts),
            SpecialTokens.end_target,
        ]
    )


def is_dialogue_in_domain(dialogue_services: list[str], domains: list[str]) -> bool:
    return all(ds in domains for ds in dialogue_services)


def delexicalize_utterance(turn: DstcTurn, schemas: dict[str, DstcSchema]) -> str:
    delexicalized_utterance = turn.utterance
    for frame in turn.frames:
        schema = schemas[frame.short_service]
        for action in frame.actions:
            for value in action.values:
                slot = next(
                    (slot for slot in schema.slots if slot.name == action.slot),
                    None,
                )
                if not slot:
                    continue
                replacement = (
                    # f"<{frame.short_service}_{humps.camelize(action.slot)}>"
                    f"<{frame.short_service}{ZsTodConstants.DOMAIN_SLOT_SEPARATOR}{action.slot}>"
                )
                delexicalized_utterance = delexicalized_utterance.replace(
                    value, replacement
                )
    return delexicalized_utterance


def get_dialog_studio_step_data(step_name: str, dataset: any):
    if step_name == Steps.DEV.value:
        step_name = "validation"
    return dataset[step_name]
