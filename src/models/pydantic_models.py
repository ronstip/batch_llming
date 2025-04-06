from typing import Literal, Dict, Any, Type, List, Optional
from pydantic import BaseModel, Field, create_model

def create_dynamic_model(name: str, fields: list[dict[str, Any]]) -> Type[BaseModel]:
    """
    Create a dynamic Pydantic model using langchain's pydantic_v1.

    Args:
        name: Name of the model.
        fields: A list of dictionaries, each with 'key', 'type', and 'description'.
                If type is 'enum', provide 'options' as a list of values.

    Returns:
        A dynamically created Pydantic model class.
    """
    annotations: Dict[str, tuple] = {}

    for field in fields:
        key = field['key']
        field_type = field['type']
        description = field['description']

        if field_type == "str":
            annotations[key] = (str, Field(..., description=description))
        elif field_type == "int":
            annotations[key] = (int, Field(..., description=description))
        elif field_type == "float":
            annotations[key] = (float, Field(..., description=description))
        elif field_type == 'enum':
            options = field.get('options')
            if not options:
                raise ValueError(f"Enum field '{key}' must have 'options'.")
            annotations[key] = (Literal[tuple(options)], Field(..., description=description))
        else:
            raise TypeError(f"Unsupported type: {field_type}")

    return create_model(name, **annotations) 