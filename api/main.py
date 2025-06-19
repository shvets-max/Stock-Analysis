from fastapi import FastAPI

from scoring import dynamic_scoring, screen_stocks, static_scoring

app = FastAPI()


@app.get("/screen_stocks/")
async def run_screen_stocks():
    return screen_stocks.main()


@app.get("/static_scoring/")
async def run_static_scoring():
    return static_scoring.main()


@app.get("/dynamic_scoring/")
async def run_dynamic_scoring():
    return dynamic_scoring.main()
