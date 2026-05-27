from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rag import YCAdvisor
from src.evaluator import StartupEvaluator

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

advisor = YCAdvisor()
evaluator = StartupEvaluator()


class AskRequest(BaseModel):
    question: str
    startup_context: str = ""


class EvalRequest(BaseModel):
    one_liner: str
    problem: str
    market_size: str
    traction: str
    team_size: str
    background: str
    working_how_long: str
    why_now: str
    biggest_risk: str
    yc_batch: str = ""


@app.post("/api/ask")
async def ask(req: AskRequest):
    result = advisor.ask_with_sources(req.question)
    return result


@app.post("/api/evaluate")
async def evaluate(req: EvalRequest):
    profile = req.dict()
    result = evaluator.evaluate(profile)
    return result


@app.post("/api/verdict")
async def verdict(req: EvalRequest):
    from src.verdict import generate_verdict_rag

    profile = req.dict()
    result = generate_verdict_rag(profile, advisor)
    return result


@app.get("/api/companies")
async def companies(search: str = "", batch: str = "", limit: int = 50):
    results = advisor.search_companies(search, batch, limit)
    return {"companies": results}


@app.get("/api/benchmark")
async def benchmark():
    from src.evaluator import run_benchmark

    results = run_benchmark(advisor)
    return results


@app.post("/api/benchmark")
async def benchmark_post():
    from src.evaluator import run_benchmark

    results = run_benchmark(advisor)
    return results
