from pydantic import BaseModel

class APIKeyCreate(BaseModel):
    label: str

class APIKeyResponse(BaseModel):
    id: int
    label: str
    key: str
    is_active: bool

    class Config:
        from_attributes = True