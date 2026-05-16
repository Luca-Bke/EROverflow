import json

import pytest

from test_agent import send_text_message, validate_event
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False])  # allows for continous polling or something,
# but really messes up the test output, so we'll just test non-streaming for now
async def test_openrouter_basic_nst(agent, streaming):

    complaint = json.dumps({
      "chief_complaint": "My wrist's been hurting since yesterday after I tripped. It's not too bad, but still hurts.",
      "vitals": {
        "hr": 82,
        "bp_sys": 128,
        "bp_dia": 75,
        "spo2": 98,
        "rr": 16,
        "temp": 36.8,
        "avpu": "A"
      },
      "history": {
        "age": 29,
        "gender": "female",
        "relevant PMH": "none",
        "time course": "pain started 24 hours ago after fall"
      }
    })

    cat = 4
    events = await send_text_message(complaint, agent, streaming=streaming)

    all_errors = []

    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)
                print(f"Received message event: {msg.model_dump()}")

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                print(f"Received task event: {json.dumps(task.model_dump(), indent=2)}")
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)
