from langchain_core.messages import AIMessage

import stt
from llm_agent import process_user_prompt
from tts import text_to_speech
from stt import process

stt.TRIGGER_WORD = "jennifer"

_printed = set()


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def to_llm(input: str):
    trigger_word_removed = input.replace(stt.TRIGGER_WORD, "")
    events = process_user_prompt(trigger_word_removed)
    for event in events:
        _print_event(event, _printed)
        message = event.get("messages")[-1]
        if (isinstance(message, AIMessage) and message.content):
            print("\n>>> Last AIMessage : " + message.content)
            text_to_speech(message.content)


process(to_llm)
