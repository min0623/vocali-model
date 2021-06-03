from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from starlette_context import context, plugins
from starlette_context.middleware import ContextMiddleware

app = FastAPI()
app.some_var = "11"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
 )
app.add_middleware(ContextMiddleware)

@app.on_event("startup")
def startup():
    context.update(test='test')

@app.get('/')
def index():
    context.update(test='test2')
    return context['test']