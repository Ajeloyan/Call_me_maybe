from pydantic import BaseModel
from typing import Dict


class PromptItem(BaseModel):
    prompt: str


class TypeInfo(BaseModel):
    type: str


class FuncItems(BaseModel):
    name: str
    description: str
    parameters: Dict[str, TypeInfo]
    returns: TypeInfo
