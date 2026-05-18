from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
  return {"app": "App working successfully"}
