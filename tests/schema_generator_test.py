import pytest
from typing import List, Optional, Union, Annotated
from pydantic import BaseModel, Field
from llm_easy_tools import get_function_schema, insert_prefix, LLMFunction
from llm_easy_tools.schema_generator import parameters_basemodel_from_function, get_name, get_tool_defs

def simple_function(count: int, size: Optional[float] = None):
    """Simple function does something."""
    pass

def simple_function_no_docstring(apple: Annotated[str, 'The apple'], banana: Annotated[str, 'The banana']):
    pass

def test_function_schema():
    schema = get_function_schema(simple_function)
    assert schema['name'] == 'simple_function'
    assert schema['description'] == 'Simple function does something.'
    params = schema['parameters']
    assert len(params['properties']) == 2
    assert params['type'] == "object"
    assert params['properties']['count']['type'] == "integer"
    assert 'size' in params['properties']
    assert 'title' not in params
    assert 'title' not in params['properties']['count']
    assert 'description' not in params

def test_noparams():
    def function_with_no_params():
        """This function has a docstring and takes no parameters."""
        pass

    def function_no_doc():
        pass

    result = get_function_schema(function_with_no_params)
    assert result['name'] == 'function_with_no_params'
    assert result['description'] == "This function has a docstring and takes no parameters."
    assert result['parameters']['properties'] == {}

    result = get_function_schema(function_no_doc)
    assert result['name'] == 'function_no_doc'
    assert result['description'] == ''
    assert result['parameters']['properties'] == {}

def test_nested():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        """Some Bar"""
        apple: str = Field(description="The apple")
        banana: str = Field(description="The banana")

    class FooAndBar(BaseModel):
        foo: Foo
        bar: Bar

    def nested_structure_function(foo: Foo, bars: List[Bar]):
        """Spams everything."""
        pass

    schema = get_function_schema(nested_structure_function)
    assert schema['name'] == 'nested_structure_function'
    assert schema['description'] == 'Spams everything.'
    assert len(schema['parameters']['properties']) == 2

    schema = get_function_schema(FooAndBar)
    assert schema['name'] == 'FooAndBar'
    assert len(schema['parameters']['properties']) == 2

def test_methods():
    class ExampleClass:
        def simple_method(self, count: int, size: Optional[float] = None):
            """Simple method does something."""
            pass

    example = ExampleClass()
    schema = get_function_schema(example.simple_method)
    assert schema['name'] == 'simple_method'
    assert schema['description'] == 'Simple method does something.'
    params = schema['parameters']
    assert len(params['properties']) == 2

def test_LLMFunction():
    def new_simple_function(count: int, size: Optional[float] = None):
        """Simple function does something."""
        pass

    func = LLMFunction(new_simple_function, name='changed_name')
    schema = func.schema
    assert schema['name'] == 'changed_name'
    assert not schema.get('strict', False)

    func = LLMFunction(simple_function, strict=True)
    schema = func.schema
    assert schema['strict'] == True

def test_merge_schemas():
    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Was the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, schema)
    assert new_schema['name'] == "Reflection_and_simple_function"
    assert len(new_schema['parameters']['properties']) == 4
    assert len(new_schema['parameters']['required']) == 3
    assert len(schema['parameters']['properties']) == 2
    assert len(schema['parameters']['required']) == 1
    param_names = list(new_schema['parameters']['properties'].keys())
    assert param_names == ['relevancy', 'next_actions_plan', 'count', 'size']

    schema = get_function_schema(simple_function)
    new_schema = insert_prefix(Reflection, schema, case_insensitive=True)
    assert new_schema['name'] == "reflection_and_simple_function"

def test_noparams_function_merge():
    def function_no_params():
        pass

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Was the last retrieved information relevant and why?")
        next_actions_plan: str = Field(..., description="What you plan to do next and why")

    schema = get_function_schema(function_no_params)
    assert schema['name'] == 'function_no_params'
    assert schema['parameters']['properties'] == {}

    new_schema = insert_prefix(Reflection, schema)
    assert len(new_schema['parameters']['properties']) == 2
    assert new_schema['name'] == 'Reflection_and_function_no_params'

def test_model_init_function():
    class User(BaseModel):
        """A user object."""
        name: str
        city: str

    schema = get_function_schema(User)
    assert schema['name'] == 'User'
    assert schema['description'] == 'A user object.'
    assert len(schema['parameters']['properties']) == 2
    assert len(schema['parameters']['required']) == 2

    new_func = LLMFunction(User, name="extract_user_details")
    assert new_func.schema['name'] == 'extract_user_details'
    assert new_func.schema['description'] == 'A user object.'
    assert len(new_func.schema['parameters']['properties']) == 2
    assert len(new_func.schema['parameters']['required']) == 2

def test_case_insensitivity():
    class User(BaseModel):
        """A user object."""
        name: str
        city: str

    schema = get_function_schema(User, case_insensitive=True)
    assert schema['name'] == 'user'
    assert get_name(User, case_insensitive=True) == 'user'

def test_function_no_type_annotation():
    def function_with_missing_type(param):
        return f"Value is {param}"

    with pytest.raises(ValueError) as exc_info:
        get_function_schema(function_with_missing_type)
    assert str(exc_info.value) == "Parameter 'param' has no type annotation"

def test_pydantic_param():
    class Query(BaseModel):
        query: str
        region: str

    def search(query: Query):
        ...

    schema = get_tool_defs([search])
    assert schema[0]['function']['name'] == 'search'
    assert schema[0]['function']['description'] == ''
    assert schema[0]['function']['parameters']['properties']['query']['$ref'] == '#/$defs/Query'

def test_strict():
    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        speciality: str
        addresses: list[Address]

    def print_companies(companies: list[Company]):
        ...

    schema = get_tool_defs([print_companies], strict=True)
    pprint(schema)
    func_schema = schema[0]['function']
    assert func_schema['name'] == 'print_companies'
    assert func_schema['strict'] == True
    assert func_schema['parameters']['additionalProperties'] == False
    assert func_schema['parameters']['$defs']['Address']['additionalProperties'] == False
    assert func_schema['parameters']['$defs']['Address']['properties']['street']['type'] == 'string'
    assert func_schema['parameters']['$defs']['Company']['additionalProperties'] == False