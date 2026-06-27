from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class Item(BaseModel):
    name: str
    quantity: int


@app.post("/items")
def create_item(item: Item):
    return {"item": item.model_dump()}
