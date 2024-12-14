import uuid
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_google_vertexai import ChatVertexAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Annotated

from pyowm import OWM
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig


@dataclass
class Device:
    device_id: str
    description: str
    properties: dict[str, str] = field(default_factory=dict)
    status: str = field(default="off")


devices = [Device("tv", "Television"),
           Device("bedroom_light", "Bedroom Light", {"brightness": "100"}),
           Device("living_room_light", "Living Room Light", {"brightness": "100"}),
           Device("bedroom_ac", "Bedroom Air Conditioner", {"temp": "20"}),
           Device("livingroom_ac", "Living Room Air Conditioner", {"temp": "20"})]

device_dict = {device.device_id: device for device in devices}
user_info = dict[str, str]()


@tool
def set_user_info(property: str, value: str):
    """
    Set various settings about the user. Such as name,location..etc
    :param property:
    :param value:
    :return:
    """
    user_info[property] = value


def get_user_info(property: str) -> str:
    """
    Get various settings about the user. Such as name,location..etc
    :param property:
    :return:
    """
    return user_info.get(property, "")


@tool
def get_today_weather(location: str) -> dict[str, str]:
    """Get today's weather of location
    Args:
        location: location to get the weather info for.

    Returns:
        Weather information of the location."""

    # using weather data from https://openweathermap.org/
    owm = OWM('ef75c938cab1aa5d00489973093e8d71')
    mgr = owm.weather_manager()

    observation = mgr.weather_at_place(location)
    w = observation.weather

    r = dict()
    r['status'] = w.detailed_status
    r['wind'] = w.wind()
    r['humidity'] = w.humidity
    r['temperature'] = str(w.temperature('celsius')) + " C"
    r['rain'] = w.rain
    r['heat_index'] = w.heat_index
    r['clouds'] = w.clouds
    return r


@tool
def list_devices() -> list[Device]:
    """List available devices.

    Returns:
        a list of available devices
    """
    return list(device_dict.values())


@tool
def turn_on_device(device_id: str) -> str:
    """Turn on a device by id.

    Args:
        device_id: string ID of the device to turn on.

    Returns:
        A message indicating that the device was turned on.
    """

    if device_id not in device_dict:
        raise ValueError("Device " + device_id + " not found")

    device_dict[device_id].status = "on"
    return device_id + " turned on"


@tool
def turn_off_device(device_id: str) -> str:
    """Turn off a device by id.

    Args:
        device_id: string ID of the device to turn off.

    Returns:
        A string message indicating that the device was turned off.
    """
    if device_id not in device_dict:
        raise ValueError("Device " + device_id + " not found")
    device_dict[device_id].status = "off"
    return device_id + " turned off"


@tool
def set_device_parameter(device_id: str, parameter: str, parameter_value: str) -> str:
    """Set parameter for a device

    Args:
        device_id: target device_id
        parameter: parameter to be changed in the device
        parameter_value: new value of the parameter"""

    if device_id not in device_dict:
        raise ValueError("Device " + device_id + " not found")
    device = device_dict[device_id]
    if (parameter not in device.properties):
        raise ValueError("Device " + device_id + " doesnt have property : " + property)
    device.properties[parameter] = parameter_value
    return "Device " + device_id + " , " + parameter + " set to " + parameter_value


@tool
def get_device_status(device_id: str) -> str:
    """Get the status of a device.

    Args:
        device_id: The ID of the device to get the status of.

    Returns:
        The status of the device.
    """
    if device_id not in device_dict:
        raise ValueError("Device " + device_id + " not found")
    return device_dict[device_id].status

#### end of utils

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

########## LangGraph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("user_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                    not result.content
                    or isinstance(result.content, list)
                    and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatVertexAI(model_name="gemini-1.5-pro-001")
#llm = OllamaFunctions(model="llama3.1", format="json")
# llm = OllamaLLM(model="llama3.1")

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart home assistant helping the user. "
            "Use the provided tools to list the devices, turn off and on devices and other functionalities."
            "This is the list of available devices. Use only device_id and properties defined here {devices}."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now, devices=list(device_dict.values()))

assistant_tools = [list_devices,
                   turn_on_device,
                   turn_off_device,
                   get_device_status,
                   get_today_weather,
                   set_device_parameter,
                   set_user_info,
                   get_user_info]
assistant_runnable = assistant_prompt | llm.bind_tools(assistant_tools)

# Graph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "user_id": "ashika",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

def process_user_prompt(user_input: str):
    events = graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
    )
    return events

