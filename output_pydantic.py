from pydantic import BaseModel
from typing import List

"""
structure output from LLM using pydantic, refer to prompt for a description of the fields.
"""

class SensitiveData(BaseModel):
    line_number: int
    type_of_data: str
    description: str
    sensitive_data: str

class ResponseOutput(BaseModel):
    file_name: str
    file_decription: str
    sensitive_data_count: int
    sensitive_data_list: List[SensitiveData] = []