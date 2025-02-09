import inspect
from typing import Annotated, Callable, Dict, Any, get_origin, Type, Union
from typing_extensions import TypeGuard

import copy
import pydantic as pd
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from pprint import pprint
import sys

class LLMFunction:
    def __init__(self, func, schema=None, name=None, description=None, strict=False):
        """
        Initializes an LLMFunction instance.

        Args:
            func: The function to be wrapped.
            schema: A pre-defined schema for the function.
            name: The name of the function.
            description: The description of the function.
            strict: Whether to enforce strict schema validation.
        """
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

        if schema:
            self.schema = schema
            if name or description:
                raise ValueError("Cannot specify name or description when providing a complete schema")
        else:
            self.schema = get_function_schema(func, strict=strict)

            if name:
                self.schema['name'] = name

            if description:
                self.schema['description'] = description

    def __call__(self, *args, **kwargs):
        """
        Calls the wrapped function with the provided arguments.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        return self.func(*args, **kwargs)

def tool_def(function_schema):
    """
    Creates a tool definition dictionary from a function schema.

    Args:
        function_schema: The schema of the function.

    Returns:
        The tool definition dictionary.
    """
    return {
        "type": "function",
        "function": function_schema,
    }

def get_tool_defs(
        functions,
        case_insensitive=False,
        prefix_class=None,
        prefix_schema_name=True,
        strict=False
):
    """
    Generates tool definitions for a list of functions or LLMFunctions.

    Args:
        functions: The list of functions or LLMFunctions.
        case_insensitive: Whether to treat function names case-insensitively.
        prefix_class: A Pydantic BaseModel class to prefix the schema.
        prefix_schema_name: Whether to include the prefix class name in the schema.
        strict: Whether to enforce strict schema validation.

    Returns:
        The list of tool definitions.
    """
    result = []
    for function in functions:
        if isinstance(function, LLMFunction):
            fun_schema = function.schema
        else:
            fun_schema = get_function_schema(function, case_insensitive, strict)

        if prefix_class:
            fun_schema = insert_prefix(prefix_class, fun_schema, prefix_schema_name, case_insensitive)
        result.append(tool_def(fun_schema))
    return result

def parameters_basemodel_from_function(function):
    """
    Creates a Pydantic BaseModel from the parameters of a function.

    Args:
        function: The function to extract parameters from.

    Returns:
        The created Pydantic BaseModel.
    """
    fields = {}
    parameters = inspect.signature(function).parameters
    function_globals = sys.modules[function.__module__].__dict__ if inspect.ismethod(function) else getattr(function, '__globals__', {})

    for name, parameter in parameters.items():
        description = None
        type_ = parameter.annotation
        if type_ is inspect._empty:
            raise ValueError(f"Parameter '{name}' has no type annotation")
        if get_origin(type_) is Annotated:
            description = type_.__metadata__[0] if type_.__metadata__ else None
            type_ = type_.__args__[0]
        if isinstance(type_, str):
            type_ = eval(type_, function_globals)
        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (type_, pd.Field(default, description=description))
    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)

def _recursive_purge_titles(d):
    """
    Recursively removes 'title' keys from a dictionary.

    Args:
        d: The dictionary to process.
    """
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == 'title' and "type" in d.keys():
                del d[key]
            else:
                _recursive_purge_titles(d[key])

def get_name(func, case_insensitive=False):
    """
    Retrieves the name of a function or LLMFunction.

    Args:
        func: The function or LLMFunction.
        case_insensitive: Whether to treat the name case-insensitively.

    Returns:
        The name of the function or LLMFunction.
    """
    schema_name = func.schema['name'] if isinstance(func, LLMFunction) else func.__name__
    return schema_name.lower() if case_insensitive else schema_name

def get_function_schema(function, case_insensitive=False, strict=False):
    """
    Generates a function schema from a function or LLMFunction.

    Args:
        function: The function or LLMFunction.
        case_insensitive: Whether to treat the function name case-insensitively.
        strict: Whether to enforce strict schema validation.

    Returns:
        The generated function schema.
    """
    if isinstance(function, LLMFunction):
        if case_insensitive:
            raise ValueError("Cannot case insensitive for LLMFunction")
        return function.schema

    description = function.__doc__.strip() if function.__doc__ else ''
    schema_name = get_name(function, case_insensitive)

    function_schema = {
        'name': schema_name,
        'description': description,
    }
    model = parameters_basemodel_from_function(function)
    model_json_schema = model.model_json_schema()
    function_schema['parameters'] = to_strict_json_schema(model_json_schema) if strict else model_json_schema
    if strict:
        function_schema['strict'] = True
    else:
        _recursive_purge_titles(function_schema['parameters'])

    return function_schema

def to_strict_json_schema(schema):
    """
    Converts a JSON schema to a strict JSON schema.

    Args:
        schema: The JSON schema to convert.

    Returns:
        The strict JSON schema.
    """
    return _ensure_strict_json_schema(schema, ())

def _ensure_strict_json_schema(json_schema, path):
    """
    Ensures a JSON schema conforms to the strict standard.

    Args:
        json_schema: The JSON schema to process.
        path: The path to the current location in the schema.

    Returns:
        The processed JSON schema.
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {key: _ensure_strict_json_schema(prop_schema, (*path, "properties", key)) for key, prop_schema in properties.items()}

    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, (*path, "items"))

    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [_ensure_strict_json_schema(variant, (*path, "anyOf", str(i))) for i, variant in enumerate(any_of)]

    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        json_schema["allOf"] = [_ensure_strict_json_schema(entry, (*path, "allOf", str(i))) for i, entry in enumerate(all_of)]

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, (*path, "$defs", def_name))

    return json_schema

def is_dict(obj):
    """
    Checks if an object is a dictionary.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a dictionary, False otherwise.
    """
    return isinstance(obj, dict)

def insert_prefix(prefix_class, schema, prefix_schema_name=True, case_insensitive=False):
    """
    Inserts a prefix class schema into a function schema.

    Args:
        prefix_class: The Pydantic BaseModel class to prefix.
        schema: The function schema.
        prefix_schema_name: Whether to include the prefix class name in the schema.
        case_insensitive: Whether to treat the prefix class name case-insensitively.

    Returns:
        The updated function schema.
    """
    if not issubclass(prefix_class, BaseModel):
        raise TypeError("The given class reference is not a subclass of pydantic BaseModel")
    
    prefix_schema = prefix_class.model_json_schema()
    _recursive_purge_titles(prefix_schema)
    prefix_schema.pop('description', '')

    if 'parameters' in schema:
        prefix_schema['required'].extend(schema['parameters'].get('required', []))
        prefix_schema['properties'].update(schema['parameters']['properties'])

    new_schema = schema.copy()
    new_schema['parameters'] = prefix_schema
    if not prefix_schema['properties']:
        new_schema.pop('parameters')
    if prefix_schema_name:
        prefix_name = get_name(prefix_class, case_insensitive)
        new_schema['name'] = f"{prefix_name}_and_{schema['name']}"

    return new_schema

if __name__ == "__main__":
    def function_with_doc():
        """This function has a docstring and no parameters."""
        pass

    altered_function = LLMFunction(function_with_doc, name="altered_name")

    class ExampleClass:
        def simple_method(self, count: int, size: float):
            """simple method does something"""
            pass

    example_object = ExampleClass()

    class User(BaseModel):
        name: str
        age: int

    pprint(get_tool_defs([
        example_object.simple_method, 
        function_with_doc, 
        altered_function,
        User
    ]))