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
        self.func = func
        self.name = name if name else func.__name__
        self.description = description if description else func.__doc__
        self.module = func.__module__

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
        return self.func(*args, **kwargs)


def tool_def(function_schema: dict) -> dict:
    return {
        "type": "function",
        "function": function_schema,
    }

def get_tool_defs(
        functions: list[Union[Callable, LLMFunction]],
        case_insensitive: bool = False,
        prefix_class: Union[Type[BaseModel], None] = None,
        prefix_schema_name: bool = True,
        strict: bool = False
        ) -> list[dict]:
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

def parameters_basemodel_from_function(function: Callable) -> Type[pd.BaseModel]:
    fields = {}
    parameters = inspect.signature(function).parameters
    # Get the global namespace, handling both functions and methods
    if inspect.ismethod(function):
        # For methods, get the class's module globals\n        function_globals = sys.modules[function.__module__].__dict__\n    else:\n        # For regular functions, use __globals__ if available\n        function_globals = getattr(function, '__globals__', {})\n\n    for name, parameter in parameters.items():\n        description = None\n        type_ = parameter.annotation\n        if type_ is inspect._empty:\n            raise ValueError(f"Parameter '{name}' has no type annotation")\n        if get_origin(type_) is Annotated:\n            if type_.__metadata__:\n                description = type_.__metadata__[0]\n            type_ = type_.__args__[0]\n        if isinstance(type_, str):\n            # this happens in postponed annotation evaluation, we need to try to resolve the type\n            # if the type is not in the global namespace, we will get a NameError\n            type_ = eval(type_, function_globals)\n        default = PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default\n        fields[name] = (type_, pd.Field(default, description=description))\n    return pd.create_model(f'{function.__name__}_ParameterModel', **fields)\n\n\ndef _recursive_purge_titles(d: Dict[str, Any]) -> None:\n    """Remove a titles from a schema recursively"""\n    if isinstance(d, dict):\n        for key in list(d.keys()):\n            if key == 'title' and "type" in d.keys():\n                del d[key]\n            else:\n                _recursive_purge_titles(d[key])\n\ndef get_name(func: Union[Callable, LLMFunction], case_insensitive: bool = False) -> str:\n    if isinstance(func, LLMFunction):\n        schema_name = func.name\n    else:\n        schema_name = func.__name__\n\n    if case_insensitive:\n        schema_name = schema_name.lower()\n    return schema_name\n\ndef get_function_schema(function: Union[Callable, LLMFunction], case_insensitive: bool=False, strict: bool=False) -> dict:\n    if isinstance(function, LLMFunction):\n        if case_insensitive:\n            raise ValueError("Cannot case insensitive for LLMFunction")\n        return function.schema\n\n    description = ''\n    if hasattr(function, '__doc__') and function.__doc__:\n        description = function.__doc__\n\n    schema_name = function.__name__\n    if case_insensitive:\n        schema_name = schema_name.lower()\n\n    function_schema: dict[str, Any] = {\n        'name': schema_name,\n        'description': description.strip(),\n    }\n    model = parameters_basemodel_from_function(function)\n    model_json_schema = model.model_json_schema()\n    if strict:\n        model_json_schema = to_strict_json_schema(model_json_schema)\n        function_schema['strict'] = True\n    else:\n        _recursive_purge_titles(model_json_schema)\n    function_schema['parameters'] = model_json_schema\n\n    return function_schema\n\n# copied from openai implementation which also uses Apache 2.0 license\n\ndef to_strict_json_schema(schema: dict) -> dict[str, Any]:\n    return _ensure_strict_json_schema(schema, path=())\n\ndef _ensure_strict_json_schema(\n    json_schema: object,\n    path: tuple[str, ...],\n) -> dict[str, Any]:\n    """Mutates the given JSON schema to ensure it conforms to the `strict` standard\n    that the API expects.\n    """\n    if not is_dict(json_schema):\n        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")\n\n    typ = json_schema.get("type")\n    if typ == "object" and "additionalProperties" not in json_schema:\n        json_schema["additionalProperties"] = False\n\n    # object types\n    # { 'type': 'object', 'properties': { 'a':  {...} } }\n    properties = json_schema.get("properties")\n    if is_dict(properties):\n        json_schema["required"] = [prop for prop in properties.keys()]\n        json_schema["properties"] = {\n            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key))\n            for key, prop_schema in properties.items()\n        }\n\n    # arrays\n    # { 'type': 'array', 'items': {...} }\n    items = json_schema.get("items")\n    if is_dict(items):\n        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"))\n\n    # unions\n    any_of = json_schema.get("anyOf")\n    if isinstance(any_of, list):\n        json_schema["anyOf"] = [\n            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i))) for i, variant in enumerate(any_of)\n        ]\n\n    # intersections\n    all_of = json_schema.get("allOf")\n    if isinstance(all_of, list):\n        json_schema["allOf"] = [\n            _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i))) for i, entry in enumerate(all_of)\n        ]\n\n    defs = json_schema.get("$defs")\n    if is_dict(defs):\n        for def_name, def_schema in defs.items():\n            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))\n\n    return json_schema\n\n\ndef is_dict(obj: object) -> TypeGuard[dict[str, object]]:\n    # just pretend that we know there are only `str` keys\n    # as that check is not worth the performance cost\n    return isinstance(obj, dict)\n\ndef insert_prefix(prefix_class, schema, prefix_schema_name=True, case_insensitive = False):\n    if not issubclass(prefix_class, BaseModel):\n        raise TypeError(\n            f"The given class reference is not a subclass of pydantic BaseModel"\n        )\n    prefix_schema = prefix_class.model_json_schema()\n    _recursive_purge_titles(prefix_schema)\n    prefix_schema.pop('description', '')\n\n    if 'parameters' in schema:\n        required = schema['parameters'].get('required', [])\n        prefix_schema['required'].extend(required)\n        for key, value in schema['parameters']['properties'].items():\n            prefix_schema['properties'][key] = value\n    new_schema = copy.copy(schema)  # Create a shallow copy of the schema\n    new_schema['parameters'] = prefix_schema\n    if len(new_schema['parameters']['properties']) == 0:  # If the parameters list is empty\n        new_schema.pop('parameters')\n    if prefix_schema_name:\n        if case_insensitive:\n            prefix_name = prefix_class.__name__.lower()\n        else:\n            prefix_name = prefix_class.__name__\n        new_schema['name'] = prefix_name + "_and_" + schema['name']\n    return new_schema\n\n\n#######################################\n#\n# Examples\n\nif __name__ == "__main__":\n    def function_with_doc():\n        """\n        This function has a docstring and no parameteres.\n        Expected Cost: high\n        """\n        pass\n\n    altered_function = LLMFunction(function_with_doc, name="altered_name")\n\n    class ExampleClass:\n        def simple_method(self, count: int, size: float):\n            """simple method does something"""\n            pass\n\n    example_object = ExampleClass()\n\n    class User(BaseModel):\n        name: str\n        age: int\n\n    pprint(get_tool_defs([\n        example_object.simple_method, \n        function_with_doc, \n        altered_function,\n        User\n        ]))