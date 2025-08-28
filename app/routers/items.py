from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field


class Item(BaseModel):
    """アイテムの基本情報"""

    name: str = Field(..., description="アイテム名")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "Plumbus"},
                {"name": "Portal Gun"},
            ]
        }
    }


class ItemWithId(Item):
    """アイテムの詳細情報（ID を含む）"""

    item_id: str = Field(..., description="アイテムの識別子（Path パラメータ）")


class ErrorResponse(BaseModel):
    """エラーレスポンスの共通形式"""

    detail: str = Field(..., description="エラーメッセージの詳細")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"detail": "Item not found"},
                {"detail": "You can only update the item: plumbus"},
            ]
        }
    }


router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}},
)
fake_items_db: dict[str, Item] = {"plumbus": Item(name="Plumbus"), "gun": Item(name="Portal Gun")}


@router.get(
    "/",
    summary="アイテム一覧の取得",
    description=("登録されているすべてのアイテムを返します。返却値はキーが item_id、値が Item モデルのオブジェクトである連想配列です。"),
    response_model=dict[str, Item],
    response_description="キーが item_id、値が Item モデルのオブジェクト",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "plumbus": {"name": "Plumbus"},
                        "gun": {"name": "Portal Gun"},
                    }
                }
            }
        }
    },
)
async def read_items() -> dict[str, Item]:
    """アイテムを全件取得します。"""
    return fake_items_db


@router.get(
    "/{item_id}",
    summary="アイテム詳細の取得",
    description="指定した item_id のアイテムを返します。存在しない場合は 404 を返します。",
    response_model=ItemWithId,
    response_description="指定した item_id のアイテム詳細",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "指定したアイテムが存在しません",
            "content": {"application/json": {"example": {"detail": "Item not found"}}},
        }
    },
)
async def read_item(item_id: str = Path(..., description="取得するアイテムID", example="plumbus")) -> ItemWithId:
    """単一アイテムを取得します。"""
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemWithId(name=fake_items_db[item_id].name, item_id=item_id)


@router.put(
    "/{item_id}",
    tags=["custom"],
    summary="アイテムの更新（サンプル）",
    description="plumbus のみ更新可能なサンプル実装です。それ以外は 403 を返します。",
    response_model=ItemWithId,
    response_description="更新後のアイテム",
    responses={
        403: {
            "model": ErrorResponse,
            "description": "plumbus 以外の更新は禁止されています",
            "content": {"application/json": {"example": {"detail": "You can only update the item: plumbus"}}},
        }
    },
)
async def update_item(item_id: str = Path(..., description="更新するアイテムID", example="plumbus")) -> ItemWithId:
    """アイテムを更新します（デモ用の制限あり）"""
    if item_id != "plumbus":
        raise HTTPException(status_code=403, detail="You can only update the item: plumbus")
    return ItemWithId(item_id=item_id, name="The great Plumbus")
