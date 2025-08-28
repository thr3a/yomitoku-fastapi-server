```python
from fastapi import FastAPI, Query
from models import MsgPayload
from pydantic import BaseModel, Field
from typing import Annotated

messages_list: dict[int, MsgPayload] = {}


class RootResponse(BaseModel):
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "Hello"}, {"message": "Welcome to the API"}]
        }
    }


class ErrorResponse(BaseModel):
    detail: str

    model_config = {"json_schema_extra": {"examples": [{"detail": "Invalid request"}]}}


@app.get(
    path="/{path}",
    summary="Root endpoint",
    description="This is the root endpoint of the API.",
    tags=["Root"],
    status_code=200,
    response_description="A simple greeting message",
    responses={
        202: {
            "description": "Accepted",
            "content": {
                "application/json": {"example": {"message": "Hello"}},
                "application/xml": {"example": {"message": "Hello"}},
            },
        },
        400: {
            "model": ErrorResponse,
        },
        404: {
            "description": "Not Found",
            "content": {"application/json": {"example": {"detail": "Item not found"}}},
        },
    },
)
def root(id: int) -> RootResponse:
    """
    Root endpoint that returns a simple greeting message.
    """
    return RootResponse(message="Hello")


class MsgParams(BaseModel):
    id: int | None = Field(
        default=None, description="The ID of the message to retrieve", ge=0
    )


@app.get("/message/")
def get_by_id(param: Annotated[MsgParams, Query()]) -> dict[str, MsgPayload]:
    """
    Helper function to get a message by its ID.
    """
    if param.id in messages_list:
        return messages_list[param.id]
    else:
        raise ValueError("Message not found")


# About page route
@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the about page."}


# Route to add a message
@app.post("/messages/{msg_name}/")
def add_msg(msg_name: str) -> dict[str, MsgPayload]:
    # Generate an ID for the item based on the highest ID in the messages_list
    msg_id = max(messages_list.keys()) + 1 if messages_list else 0
    messages_list[msg_id] = MsgPayload(msg_id=msg_id, msg_name=msg_name)

    return {"message": messages_list[msg_id]}


class MsgBody(BaseModel):
    msg_name: str = Field(..., description="The name of the message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"msg_name": "Hello"},
            ]
        }
    }


# Route to add a message
@app.post("/messages/")
def add_msg_body(request: MsgBody) -> dict[str, MsgPayload]:
    # Generate an ID for the item based on the highest ID in the messages_list
    msg_id = max(messages_list.keys()) + 1 if messages_list else 0
    messages_list[msg_id] = MsgPayload(msg_id=msg_id, msg_name=request.msg_name)

    return {"message": messages_list[msg_id]}


# Route to list all messages
@app.get("/messages")
def message_items() -> dict[str, dict[int, MsgPayload]]:
    return {"messages:": messages_list}
```

# 参考リンク

- [FastAPIのOpenAPIの出力メモ](https://zenn.dev/team_delta/articles/a54b32202b9252)
