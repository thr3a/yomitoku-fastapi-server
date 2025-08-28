"""yomitoku を HTTP 経由で提供する FastAPI アプリ。

このモジュールは画像/PDF を受け取り OCR/レイアウト解析を行う `/analyze`
エンドポイントを提供します。出力形式は `json` / `markdown` / `vertical` /
`horizontal` に対応しています。実装はリポジトリ内の `example.py` を参考に、
プロジェクトの Ruff ルールに準拠しています。
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_pdf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

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


@app.get("/")
async def root() -> dict[str, str]:
    """トップページ。"""
    return {"message": "turai.work"}


@app.get("/health")
async def health() -> dict[str, str]:
    """ヘルスチェック用エンドポイント。"""
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    """アプリ起動時のログ。"""
    logger.info("FastAPI サーバーの準備ができました。リクエストを受け付けます。")


@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(..., description="入力ファイル（画像またはPDF)"),
    format: Literal["json", "markdown", "vertical", "horizontal"] = Query(
        "json",
        description="出力フォーマット (json, markdown, vertical, horizontal)",
    ),
) -> dict[str, object]:
    """画像/PDF からテキスト・レイアウトを抽出するエンドポイント。

    - json: yomitoku の生 JSON（ページごと）
    - markdown: 解析結果を Markdown に変換して連結
    - vertical: 縦書きテキストのみを抽出して連結
    - horizontal: 横書きテキストのみを抽出して連結
    """
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
            return {"format": "markdown", "content": final_markdown}

        if fmt in {"vertical", "horizontal"}:
            all_text: list[str] = []
            for results in all_results:
                json_content = json.loads(results.model_dump_json())
                text = "\n".join(p["contents"] for p in json_content["paragraphs"] if p["direction"] == fmt and p["contents"] is not None)
                all_text.append(text)
            return {"format": fmt, "content": "\n\n".join(all_text)}

        # 既定: ページ単位の生 JSON を返す
        json_results = [json.loads(results.model_dump_json()) for results in all_results]
        return {"format": "json", "content": json_results}
    finally:
        temp_path.unlink(missing_ok=True)
