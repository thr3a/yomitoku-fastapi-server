from fastapi import FastAPI

from .routers import items

app = FastAPI()

app.include_router(items.router)


@app.get("/")
async def root() -> dict[str, str]:
    """トップページ"""
    return {"message": "turai.work"}


@app.get("/health")
async def health() -> dict[str, str]:
    """ヘルスチェック用エンドポイント。"""
    return {"status": "ok"}
