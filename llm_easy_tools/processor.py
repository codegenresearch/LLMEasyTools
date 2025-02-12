import json
from typing import Callable, Union, Optional, Any, get_origin, get_args
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError
from llm_easy_tools.schema_generator import get_name, parameters_basemodel_from_function, LLMFunction
from llm_easy_tools.types import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionMessage

class NoMatchingTool(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    output: Optional[Any] = None
    arguments: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    stack_trace: Optional[str] = None
    soft_errors: list[Exception] = field(default_factory=list)
    prefix: Optional[BaseModel] = None
    tool: Optional[Union[Callable, BaseModel]] = None

    def to_message(self) -> dict[str, str]:
        if self.error is not None:
            content = f"{self.error}"
        elif self.output is None:
            content = ''
        elif isinstance(self.output, BaseModel):
            content = f"{self.name} created"
        else:
            content = str(self.output)
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": content,
        }

def process_tool_call(tool_call, functions_or_models, prefix_class=None, fix_json_args=True, case_insensitive=False) -> ToolResult:
    function_call = tool_call.function
    tool_name = function_call.name
    args = function_call.arguments
    soft_errors: list[Exception] = []
    error = None
    stack_trace = None
    prefix = None
    output = None
    try:
        tool_args = json.loads(args)
    except json.decoder.JSONDecodeError as e:
        if fix_json_args:
            soft_errors.append(e)
            args = args.replace(', }', '}').replace(',}', '}')
            tool_args = json.loads(args)
        else:
            stack_trace = traceback.format_exc()
            return ToolResult(tool_call_id=tool_call.id, name=tool_name, error=e, stack_trace=stack_trace)

    if prefix_class is not None:
        try:
            prefix = _extract_prefix_unpacked(tool_args, prefix_class)
        except ValidationError as e:
            soft_errors.append(e)
        prefix_name = prefix_class.__name__
        if case_insensitive:
            prefix_name = prefix_name.lower()
        if not tool_name.startswith(prefix_name):
            soft_errors.append(NoMatchingTool(f"Trying to decode function call with a name '{tool_name}' not matching prefix '{prefix_name}'"))
        else:
            tool_name = tool_name[len(prefix_name + '_and_'):]

    tool = None

    for f in functions_or_models:
        if get_name(f, case_insensitive=case_insensitive) == tool_name:
            tool = f
            try:
                output, new_soft_errors = _process_unpacked(f, tool_args, fix_json_args=fix_json_args)
                soft_errors.extend(new_soft_errors)
            except Exception as e:
                error = e
                stack_trace = traceback.format_exc()
            break
    else:
        error = NoMatchingTool(f"Function {tool_name} not found")
    result = ToolResult(
        tool_call_id=tool_call.id, 
        name=tool_name,
        arguments=tool_args,
        output=output, 
        error=error,
        stack_trace=stack_trace,
        soft_errors=soft_errors,
        prefix=prefix,
        tool=tool,
    )
    return result

def split_string_to_list(s: str) -> list[str]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return [item.strip() for item in s.split(',')]

def _process_unpacked(function, tool_args={}, fix_json_args=True):
    if isinstance(function, LLMFunction):
        function = function.func
    model = parameters_basemodel_from_function(function)
    soft_errors = []
    if fix_json_args:
        for field, field_info in model.model_fields.items():
            field_annotation = field_info.annotation
            if _is_list_type(field_annotation):
                if field in tool_args and isinstance(tool_args[field], str):
                    tool_args[field] = split_string_to_list(tool_args[field])
                    soft_errors.append(f"Fixed JSON decode error for field {field}")

    model_instance = model(**tool_args)
    args = {}
    for field, _ in model.model_fields.items():
        args[field] = getattr(model_instance, field)
    return function(**args), soft_errors

def _is_list_type(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is list:
        return True
    elif origin is Union or origin is Optional:
        return any(_is_list_type(arg) for arg in args)
    return False

def _extract_prefix_unpacked(tool_args, prefix_class):
    prefix_args = {}
    for key in list(tool_args.keys()):
        if key in prefix_class.__annotations__:
            prefix_args[key] = tool_args.pop(key)
    prefix = prefix_class(**prefix_args)
    return prefix

def process_response(response: ChatCompletion, functions: list[Union[Callable, LLMFunction]], choice_num=0, **kwargs) -> list[ToolResult]:
    message = response.choices[choice_num].message
    return process_message(message, functions, **kwargs)

def process_message(
    message: ChatCompletionMessage,
    functions: list[Union[Callable, LLMFunction]],
    prefix_class=None,
    fix_json_args=True,
    case_insensitive=False,
    executor=None
    ) -> list[ToolResult]:
    results = []
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls = message.tool_calls
    else:
        tool_calls = []

    if not tool_calls:
        return []
    args_list = [(tool_call, functions, prefix_class, fix_json_args, case_insensitive) for tool_call in tool_calls]

    if executor:
        results = list(executor.map(lambda args: process_tool_call(*args), args_list))
    else:
        results = list(map(lambda args: process_tool_call(*args), args_list)) 
    return results

def process_one_tool_call(
        response: ChatCompletion,
        functions: list[Union[Callable, LLMFunction]],
        index: int = 0,
        prefix_class=None,
        fix_json_args=True,
        case_insensitive=False
    ) -> Optional[ToolResult]:
    tool_calls = _get_tool_calls(response)
    if not tool_calls or index >= len(tool_calls):
        return None

    return process_tool_call(tool_calls[index], functions, prefix_class, fix_json_args, case_insensitive)

def _get_tool_calls(response: ChatCompletion) -> list[ChatCompletionMessageToolCall]:
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        return response.choices[0].message.tool_calls
    return []