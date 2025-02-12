import pytest
from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field
from llm_easy_tools import get_function_schema, insert_prefix, LLMFunction
from llm_easy_tools.schema_generator import parameters_basemodel_from_function, get_name, get_tool_defs


def simple_func(count: int, size: Optional[float] = None):
    """Performs a simple operation."""
    pass


def simple_func_no_docs(apple: Annotated[str, 'The apple'], banana: Annotated[str, 'The banana']):
    pass


def test_func_schema():
    schema = get_function_schema(simple_func)
    assert schema['name'] == 'simple_func'
    assert schema['description'] == 'Performs a simple operation.'
    params = schema['parameters']
    assert len(params['properties']) == 2
    assert params['type'] == "object"
    assert params['properties']['count']['type'] == "integer"
    assert 'size' in params['properties']
    assert 'title' not in params
    assert 'title' not in params['properties']['count']
    assert 'description' not in params


def test_no_params():
    def func_with_no_params():
        """Function with a docstring but no parameters."""
        pass

    def func_no_docs():
        pass

    result = get_function_schema(func_with_no_params)
    assert result['name'] == 'func_with_no_params'
    assert result['description'] == "Function with a docstring but no parameters."
    assert result['parameters']['properties'] == {}

    result = get_function_schema(func_no_docs)
    assert result['name'] == 'func_no_docs'
    assert result['description'] == ''
    assert result['parameters']['properties'] == {}


def test_nested_models():
    class Foo(BaseModel):
        count: int
        size: Optional[float] = None

    class Bar(BaseModel):
        """Describes a Bar instance."""
        apple: str = Field(description="The apple")
        banana: str = Field(description="The banana")

    class FooBar(BaseModel):
        foo: Foo
        bar: Bar

    def nested_func(foo: Foo, bars: List[Bar]):
        """Processes a nested structure."""
        pass

    schema = get_function_schema(nested_func)
    assert schema['name'] == 'nested_func'
    assert schema['description'] == 'Processes a nested structure.'
    assert len(schema['parameters']['properties']) == 2

    schema = get_function_schema(FooBar)
    assert schema['name'] == 'FooBar'
    assert len(schema['parameters']['properties']) == 2


def test_methods():
    class Example:
        def method(self, count: int, size: Optional[float] = None):
            """A simple method."""
            pass

    obj = Example()

    schema = get_function_schema(obj.method)
    assert schema['name'] == 'method'
    assert schema['description'] == 'A simple method.'
    params = schema['parameters']
    assert len(params['properties']) == 2


def test_llm_function():
    def new_func(count: int, size: Optional[float] = None):
        """Performs a simple operation."""
        pass

    func = LLMFunction(new_func, name='renamed_func')
    schema = func.schema
    assert schema['name'] == 'renamed_func'
    assert not schema.get('strict', False)

    func = LLMFunction(simple_func, strict=True)
    schema = func.schema
    assert schema['strict'] is True


def test_merge_schemas():
    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Relevance and reason.")
        next_actions: str = Field(..., description="Next actions and rationale.")

    schema = get_function_schema(simple_func)
    new_schema = insert_prefix(Reflection, schema)
    assert new_schema['name'] == "Reflection_and_simple_func"
    assert len(new_schema['parameters']['properties']) == 4
    assert len(new_schema['parameters']['required']) == 3
    assert len(schema['parameters']['properties']) == 2
    assert len(schema['parameters']['required']) == 1
    param_names = list(new_schema['parameters']['properties'].keys())
    assert param_names == ['relevancy', 'next_actions', 'count', 'size']

    new_schema = insert_prefix(Reflection, schema, case_insensitive=True)
    assert new_schema['name'] == "reflection_and_simple_func"


def test_no_params_merge():
    def no_params_func():
        pass

    class Reflection(BaseModel):
        relevancy: str = Field(..., description="Relevance and reason.")
        next_actions: str = Field(..., description="Next actions and rationale.")

    schema = get_function_schema(no_params_func)
    assert schema['name'] == 'no_params_func'
    assert schema['parameters']['properties'] == {}

    new_schema = insert_prefix(Reflection, schema)
    assert len(new_schema['parameters']['properties']) == 2
    assert new_schema['name'] == 'Reflection_and_no_params_func'


def test_model_init():
    class User(BaseModel):
        """Represents a user."""
        name: str
        city: str

    schema = get_function_schema(User)
    assert schema['name'] == 'User'
    assert schema['description'] == 'Represents a user.'
    assert len(schema['parameters']['properties']) == 2
    assert len(schema['parameters']['required']) == 2

    func = LLMFunction(User, name="user_details")
    assert func.schema['name'] == 'user_details'
    assert func.schema['description'] == 'Represents a user.'
    assert len(func.schema['parameters']['properties']) == 2
    assert len(func.schema['parameters']['required']) == 2


def test_case_insensitive():
    class User(BaseModel):
        """Represents a user."""
        name: str
        city: str

    schema = get_function_schema(User, case_insensitive=True)
    assert schema['name'] == 'user'
    assert get_name(User, case_insensitive=True) == 'user'


def test_missing_type_annotation():
    def func(param):
        return f"Value: {param}"

    with pytest.raises(ValueError) as exc_info:
        get_function_schema(func)
    assert str(exc_info.value) == "Parameter 'param' has no type annotation"


def test_pydantic_param():
    class Query(BaseModel):
        query: str
        region: str

    def search(query: Query):
        pass

    schema = get_tool_defs([search])

    assert schema[0]['function']['name'] == 'search'
    assert schema[0]['function']['description'] == ''
    assert schema[0]['function']['parameters']['properties']['query']['$ref'] == '#/$defs/Query'


def test_strict_schema():
    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        speciality: str
        addresses: List[Address]

    def print_companies(companies: List[Company]):
        pass

    schema = get_tool_defs([print_companies], strict=True)

    func_schema = schema[0]['function']

    assert func_schema['name'] == 'print_companies'
    assert func_schema['strict'] is True
    assert func_schema['parameters']['additionalProperties'] is False
    assert func_schema['parameters']['$defs']['Address']['additionalProperties'] is False
    assert func_schema['parameters']['$defs']['Address']['properties']['street']['type'] == 'string'
    assert func_schema['parameters']['$defs']['Company']['additionalProperties'] is False