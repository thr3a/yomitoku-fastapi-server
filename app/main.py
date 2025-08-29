"""yomitoku を HTTP 経由で提供する FastAPI アプリ。

このモジュールは画像/PDF を受け取り OCR/レイアウト解析を行う `/analyze`
エンドポイントを提供します。出力形式は `json` / `markdown` / `vertical` /
`horizontal` に対応しています。実装はリポジトリ内の `example.py` を参考に、
プロジェクトの Ruff ルールに準拠しています。
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_pdf
from yomitoku.document_analyzer import DocumentAnalyzerSchema

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OpenAPI/Swagger UI に表示される API メタデータ
tags_metadata = [
    {
        "name": "情報",
        "description": "サービスの基本情報や動作確認に利用するエンドポイント群です。",
    },
    {
        "name": "解析",
        "description": "画像や PDF を対象に OCR・レイアウト解析を行うエンドポイント群です。",
    },
]

app = FastAPI(
    title="Yomitoku FastAPI Server",
    description=("Yomitoku を HTTP API として提供するサーバー実装です。\n\n主に以下の機能を提供します:\n- 画像/PDF のアップロードと OCR・レイアウト解析 (POST /analyze)\n- 稼働確認用のヘルスチェック (GET /health)\n\nSwagger UI では各エンドポイントの入力/出力例やエラー応答を確認できます。"),
    version="0.1.0",
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
)

# CORS の設定（開発用に緩め。本番では適切に制限してください）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# モジュール読み込み時にDocumentAnalyzerを1度だけ初期化
if not torch.cuda.is_available():
    raise SystemExit("CUDA が利用できません。GPU 環境を用意してください。")

analyzer = DocumentAnalyzer(
    configs={
        "ocr": {
            "text_detector": {
                "device": "cuda",
                "visualize": False,
            },
            "text_recognizer": {
                "device": "cuda",
                "visualize": False,
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "device": "cuda",
                "visualize": False,
            },
            "table_structure_recognizer": {
                "device": "cuda",
                "visualize": False,
            },
        },
    },
    device="cuda",
    visualize=False,
)
logger.info("DocumentAnalyzer の初期化が完了しました。")


class RootResponse(BaseModel):
    """ルートエンドポイントのレスポンス。"""

    message: str = Field(description="簡単な案内メッセージ")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "turai.work"},
                {"message": "ようこそ Yomitoku API へ"},
            ]
        }
    }


class HealthResponse(BaseModel):
    """ヘルスチェックのレスポンス。"""

    status: str = Field(description="サーバーの状態 (ok で正常)")

    model_config = {"json_schema_extra": {"examples": [{"status": "ok"}]}}


class ErrorResponse(BaseModel):
    """エラーレスポンスの共通形式。"""

    detail: str = Field(description="エラーの詳細メッセージ")

    model_config = {"json_schema_extra": {"examples": [{"detail": "画像の読み込みに失敗しました。"}]}}


@app.get(
    "/",
    summary="ルート",
    description=("API の案内を返します。動作確認用に利用できます。\n\nSwagger UI: `/docs` / ReDoc: `/redoc`"),
    tags=["情報"],
    response_model=RootResponse,
    response_description="案内メッセージ",
)
async def root() -> RootResponse:
    return RootResponse(message="turai.work")


@app.get(
    "/health",
    summary="ヘルスチェック",
    description="アプリケーションが稼働中かを確認します。監視の疎通確認にも利用できます。",
    tags=["情報"],
    response_model=HealthResponse,
    response_description="サーバーの状態",
)
async def health() -> HealthResponse:
    """ヘルスチェック用エンドポイント。"""
    return HealthResponse(status="ok")


@app.on_event("startup")
async def on_startup() -> None:
    """アプリ起動時のログ。"""
    logger.info("FastAPI サーバーの準備ができました。リクエストを受け付けます。")


class AnalyzeJsonResponse(BaseModel):
    """`format` が `json` のときのレスポンス。

    `content` はページごとの `DocumentAnalyzerSchema` の配列。
    """

    format: Literal["json"] = Field(description="返却するフォーマット種別 (常に json)")
    content: list[DocumentAnalyzerSchema] = Field(
        description="ページごとの解析結果 (構造化 JSON)",
    )


class AnalyzeMarkdownResponse(BaseModel):
    """`format` が `markdown` のときのレスポンス。"""

    format: Literal["markdown"] = Field(description="返却するフォーマット種別 (常に markdown)")
    content: str = Field(description="Markdown 文字列")


class AnalyzeVerticalResponse(BaseModel):
    """`format` が `vertical` のときのレスポンス。"""

    format: Literal["vertical"] = Field(description="返却するフォーマット種別 (常に vertical)")
    content: str = Field(description="縦書きテキストを連結した文字列")


class AnalyzeHorizontalResponse(BaseModel):
    """`format` が `horizontal` のときのレスポンス。"""

    format: Literal["horizontal"] = Field(description="返却するフォーマット種別 (常に horizontal)")
    content: str = Field(description="横書きテキストを連結した文字列")


# 判別ユニオン: `format` を判別子としてレスポンス型を切り替える
AnalyzeResponse = Annotated[
    AnalyzeJsonResponse | AnalyzeMarkdownResponse | AnalyzeVerticalResponse | AnalyzeHorizontalResponse,
    Field(discriminator="format"),
]


@app.post(
    "/analyze",
    summary="ドキュメント解析",
    description=(
        "画像 (JPG/PNG など) または PDF を受け取り、OCR とレイアウト解析を実施します。\n\n"
        "- `json`: Yomitoku の生 JSON をページ単位で返します。\n"
        "- `markdown`: 各ページの Markdown を生成し、連結して返します。\n"
        "- `vertical`/`horizontal`: 指定方向のテキストのみを抽出し、連結して返します。\n\n"
        "注意: 本サーバーは GPU(CUDA) 前提です。CUDA が無い環境では起動に失敗します。\n"
        "Content-Type は `multipart/form-data` を利用してください。"
    ),
    tags=["解析"],
    status_code=200,
    response_model=AnalyzeResponse,
    response_description="解析結果",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "無効なファイル形式や読み込み失敗などの入力エラー",
        },
        415: {
            "description": "未対応の Content-Type",
            "content": {"application/json": {"example": {"detail": "Unsupported Media Type"}}},
        },
        500: {
            "description": "サーバー内部エラー",
            "content": {"application/json": {"example": {"detail": "Internal Server Error"}}},
        },
    },
)
async def analyze_document(
    file: UploadFile = File(
        ...,
        description=("入力ファイル。画像 (JPG/PNG など) または PDF に対応。最大サイズは運用環境の制限に依存します。"),
    ),
    format: Literal["json", "markdown", "vertical", "horizontal"] = Query(
        "json",
        description=("出力フォーマット。`json`/`markdown`/`vertical`/`horizontal` から選択"),
        examples={
            "json": {"summary": "生 JSON", "value": "json"},
            "markdown": {"summary": "Markdown", "value": "markdown"},
            "vertical": {"summary": "縦書き", "value": "vertical"},
            "horizontal": {"summary": "横書き", "value": "horizontal"},
        },
    ),
) -> AnalyzeResponse:
    """画像/PDF からテキスト・レイアウトを抽出して返します。"""
    # アップロードファイルを一時ファイルとして保存（PDF や再読込の都合上必要）
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tf:
        tf.write(contents)
        temp_path = Path(tf.name)

    try:
        # 入力を 1 枚以上の画像として読み込み
        if file.filename.lower().endswith(".pdf"):
            imgs = load_pdf(str(temp_path))
        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました。対応形式のファイルを指定してください。")
            imgs = [img]

        # 画像/ページごとに解析を実行
        all_results = []
        for img in imgs:
            # analyzer.__call__ は内部で asyncio.run() を呼ぶため、
            # 非同期ルートでは run() を直接 await する。
            # ライブラリ内部の aggregate() は self.img を参照するため、
            # 非同期実行時も事前に設定しておく。
            analyzer.img = img  # type: ignore[attr-defined]
            results, _, _ = await analyzer.run(img)
            all_results.append(results)

        fmt = format.lower()
        if fmt == "markdown":
            # 一時ファイルを用いて各ページの Markdown を生成
            md_pages: list[str] = []
            for i, results in enumerate(all_results):
                with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as md_file:
                    results.to_markdown(md_file.name, img=imgs[i])
                    page_md = Path(md_file.name).read_text(encoding="utf-8")
                Path(md_file.name).unlink(missing_ok=True)
                md_pages.append(page_md)

            final_markdown = "".join((f"\n\n## Page {i + 1}\n\n" if i > 0 else "") + md for i, md in enumerate(md_pages))
            return AnalyzeResponse(format="markdown", content=final_markdown)

        if fmt in {"vertical", "horizontal"}:
            # 構造化オブジェクトから方向別にテキストを抽出
            all_text: list[str] = []
            for results in all_results:
                texts = [p.contents for p in results.paragraphs if p.direction == fmt and p.contents is not None]
                all_text.append("\n".join(texts))

            if fmt == "vertical":
                return AnalyzeVerticalResponse(format="vertical", content="\n\n".join(all_text))
            return AnalyzeHorizontalResponse(format="horizontal", content="\n\n".join(all_text))

        return AnalyzeJsonResponse(format="json", content=all_results)
    finally:
        temp_path.unlink(missing_ok=True)
