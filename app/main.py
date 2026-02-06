import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "zai-org/GLM-OCR"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = Path(
    os.getenv("GLM_MODEL_CACHE", str(ROOT_DIR / "models" / "hf_cache"))
)
DEFAULT_DPI = 220
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
ALLOWED_TASKS = {"text", "table", "formula", "extract_json"}
ALLOWED_LINEBREAK_MODES = {"none", "paragraph", "compact"}

logger = logging.getLogger("glm_ocr_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def patch_transformers_video_auto_none_bug() -> None:
    try:
        from transformers.models.auto import video_processing_auto
    except Exception:
        return

    if getattr(video_processing_auto, "_glm_none_patch_applied", False):
        return

    fixed = 0
    for key, value in list(video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES.items()):
        if value is None:
            video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES[key] = ""
            fixed += 1

    video_processing_auto._glm_none_patch_applied = True
    if fixed:
        logger.warning("Applied transformers video auto patch for %d entries", fixed)


def resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return "cuda"
    if requested == "cpu":
        return requested
    raise HTTPException(status_code=400, detail=f"Unsupported device: {device}")


class GlmRuntime:
    def __init__(self) -> None:
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForImageTextToText] = None
        self.current_device: Optional[str] = None
        self._load_lock = asyncio.Lock()

    def _load_model(self, device: str) -> AutoModelForImageTextToText:
        if device == "cuda":
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    cache_dir=str(MODEL_CACHE_DIR),
                    torch_dtype="auto",
                    device_map="auto",
                )
            except ValueError as exc:
                if "requires `accelerate`" not in str(exc):
                    raise
                logger.warning(
                    "accelerate is missing. Falling back to CUDA load without device_map."
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    cache_dir=str(MODEL_CACHE_DIR),
                    torch_dtype="auto",
                    device_map=None,
                )
                return model.to("cuda")

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float32,
            device_map=None,
        )
        return model.to("cpu")

    async def ensure_loaded(self, device: str) -> None:
        async with self._load_lock:
            if self.processor is None:
                logger.info("Loading processor: %s", MODEL_ID)
                patch_transformers_video_auto_none_bug()
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )
                except ImportError as exc:
                    if "Torchvision library" in str(exc):
                        raise RuntimeError(
                            "torchvision is required by GLM-OCR processor. "
                            "Install it with: pip install torchvision"
                        ) from exc
                    raise
                except TypeError as exc:
                    if "NoneType" not in str(exc):
                        raise
                    # Retry once after forcing the compatibility patch.
                    patch_transformers_video_auto_none_bug()
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )

            if self.model is not None and self.current_device == device:
                return

            if self.model is not None:
                logger.info(
                    "Switching model device from %s to %s", self.current_device, device
                )
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("Loading model: %s (device=%s)", MODEL_ID, device)
            self.model = await asyncio.to_thread(self._load_model, device)
            self.current_device = device

    def get(self) -> tuple[AutoProcessor, AutoModelForImageTextToText, str]:
        if self.processor is None or self.model is None or self.current_device is None:
            raise RuntimeError("GLM runtime is not initialized")
        return self.processor, self.model, self.current_device


def load_pages(path: Path, dpi: int) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "pypdfium2 is required for PDF input. Install it with: pip install pypdfium2"
            ) from exc

        pages: list[Image.Image] = []
        scale = max(36, int(dpi)) / 72.0
        doc = pdfium.PdfDocument(str(path))
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    if hasattr(bitmap, "close"):
                        bitmap.close()
                    if hasattr(page, "close"):
                        page.close()
                pages.append(image)
        finally:
            if hasattr(doc, "close"):
                doc.close()
        return pages

    with Image.open(path) as image:
        return [image.convert("RGB")]


def save_temp_upload(upload_name: str, content: bytes) -> Path:
    suffix = Path(upload_name or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def save_temp_png(image: Image.Image) -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(tmp_path, format="PNG")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def build_prompt(task: str, schema: Optional[str]) -> str:
    if task == "text":
        return "Text Recognition:"
    if task == "table":
        return "Table Recognition:"
    if task == "formula":
        return "Formula Recognition:"
    if task == "extract_json":
        if not schema:
            raise HTTPException(
                status_code=400,
                detail="schema is required when task=extract_json",
            )
        # Align with the prompt style in the official model card.
        return f"请按下列JSON格式输出图中信息:\n{schema}"
    raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")


def is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def join_soft_wrapped_line(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if is_cjk_char(left[-1]) and is_cjk_char(right[0]):
        return left + right
    return f"{left} {right}"


def is_hard_break(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left.endswith(("。", "！", "？", ".", "!", "?", "：", ":", "；", ";")):
        return True
    if "|" in left and "|" in right:
        return True
    if re.match(r"^(\d+[\.\)]|[（(]?\d+[）)]|[-*•・●○■□])\s*", right):
        return True
    return False


def normalize_linebreaks(text: str, mode: str) -> str:
    normalized_mode = (mode or "none").strip().lower()
    if normalized_mode == "none" or not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    if normalized_mode == "paragraph":
        merged: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if merged and merged[-1] != "":
                    merged.append("")
                continue
            if not merged or merged[-1] == "":
                merged.append(line)
                continue
            if is_hard_break(merged[-1], line):
                merged.append(line)
            else:
                merged[-1] = join_soft_wrapped_line(merged[-1], line)
        return "\n".join(merged).strip()

    if normalized_mode == "compact":
        non_empty = [line.strip() for line in lines if line.strip()]
        if not non_empty:
            return ""
        merged = non_empty[0]
        for line in non_empty[1:]:
            merged = join_soft_wrapped_line(merged, line)
        return merged.strip()

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported linebreak_mode: {mode}",
    )


def glm_infer(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    request_id: Optional[str] = None,
) -> tuple[str, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generation_args: dict[str, Any] = {"max_new_tokens": max(1, int(max_new_tokens))}
    if temperature is not None and float(temperature) > 0:
        generation_args.update({"do_sample": True, "temperature": float(temperature)})
    if request_id:
        generation_args["stopping_criteria"] = StoppingCriteriaList(
            [CancelStoppingCriteria(request_id)]
        )

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_args)
    input_len = inputs["input_ids"].shape[1]
    output = generated[0][input_len:]
    raw_text = processor.decode(output, skip_special_tokens=False).strip()
    clean_text = processor.decode(output, skip_special_tokens=True).strip()
    return raw_text, clean_text


RUNTIME = GlmRuntime()
GENERATE_SEMAPHORE = asyncio.Semaphore(1)
PROGRESS_STATE: dict[str, dict[str, Any]] = {}
MAX_PROGRESS_ENTRIES = 300
CANCEL_REQUESTS: set[str] = set()


def set_progress(
    request_id: str,
    state: str,
    message: str,
    current_page: int = 0,
    total_pages: int = 0,
) -> None:
    PROGRESS_STATE[request_id] = {
        "request_id": request_id,
        "state": state,
        "message": message,
        "current_page": int(current_page),
        "total_pages": int(total_pages),
        "updated_at": time.time(),
    }
    if len(PROGRESS_STATE) > MAX_PROGRESS_ENTRIES:
        # Keep memory bounded by removing the oldest entries.
        oldest = sorted(PROGRESS_STATE.items(), key=lambda item: item[1]["updated_at"])[
            : len(PROGRESS_STATE) - MAX_PROGRESS_ENTRIES
        ]
        for key, _ in oldest:
            PROGRESS_STATE.pop(key, None)


def is_cancel_requested(request_id: str) -> bool:
    return request_id in CANCEL_REQUESTS


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        return is_cancel_requested(self.request_id)


def clear_cancel_request(request_id: str) -> None:
    CANCEL_REQUESTS.discard(request_id)


def request_cancel(request_id: str) -> dict[str, Any]:
    item = PROGRESS_STATE.get(request_id)
    if item is not None:
        current_state = str(item.get("state") or "")
        if current_state in {"done", "error", "canceled"}:
            return {
                "request_id": request_id,
                "accepted": False,
                "state": current_state,
                "message": "このリクエストは既に終了しています",
            }
    CANCEL_REQUESTS.add(request_id)
    if item is None:
        return {
            "request_id": request_id,
            "accepted": True,
            "state": "cancel_requested",
            "message": "中断要求を受け付けました",
        }

    set_progress(
        request_id,
        "cancel_requested",
        "中断要求を受け付けました",
        int(item.get("current_page") or 0),
        int(item.get("total_pages") or 0),
    )
    return {
        "request_id": request_id,
        "accepted": True,
        "state": "cancel_requested",
        "message": "中断要求を受け付けました",
    }

app = FastAPI(
    title="GLM-OCR Local Server",
    description="FastAPI server for local GLM-OCR inference",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_load_model() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    default_device = resolve_device("auto")
    await RUNTIME.ensure_loaded(default_device)
    logger.info(
        "Startup complete (device=%s, cache_dir=%s)",
        default_device,
        MODEL_CACHE_DIR,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="UI not found. Ensure app/static/index.html exists.",
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_default": "cuda" if torch.cuda.is_available() else "cpu",
        "model": MODEL_ID,
        "model_cache_dir": str(MODEL_CACHE_DIR),
    }


@app.get("/api/progress/{request_id}")
async def progress(request_id: str) -> dict[str, Any]:
    item = PROGRESS_STATE.get(request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="progress not found")
    return item


@app.post("/api/cancel/{request_id}")
async def cancel(request_id: str) -> dict[str, Any]:
    return request_cancel(request_id)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    dpi: int = Form(DEFAULT_DPI),
    task: str = Form("text"),
    linebreak_mode: str = Form("none"),
    schema: Optional[str] = Form(None),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    request_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    request_id = (request_id or "").strip() or uuid.uuid4().hex
    clear_cancel_request(request_id)
    set_progress(request_id, "preprocessing", "事前処理中", 0, 0)

    normalized_task = (task or "text").strip().lower()
    if normalized_task not in ALLOWED_TASKS:
        set_progress(request_id, "error", f"Unsupported task: {task}", 0, 0)
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
    normalized_linebreak_mode = (linebreak_mode or "none").strip().lower()
    if normalized_linebreak_mode not in ALLOWED_LINEBREAK_MODES:
        set_progress(
            request_id,
            "error",
            f"Unsupported linebreak_mode: {linebreak_mode}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400, detail=f"Unsupported linebreak_mode: {linebreak_mode}"
        )

    try:
        prompt = build_prompt(normalized_task, schema)
        resolved_device = resolve_device(device)
        await RUNTIME.ensure_loaded(resolved_device)
        processor, model, actual_device = RUNTIME.get()
    except HTTPException as exc:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", str(exc.detail), 0, 0)
        raise
    except Exception as exc:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", str(exc), 0, 0)
        raise

    try:
        content = await file.read()
        input_path = save_temp_upload(file.filename or "upload.bin", content)
        pages = load_pages(input_path, dpi)
    except HTTPException:
        clear_cancel_request(request_id)
        raise
    except Exception as exc:
        logger.exception("Failed to load input file")
        clear_cancel_request(request_id)
        set_progress(request_id, "error", f"事前処理エラー: {exc}", 0, 0)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "input_path" in locals():
            Path(input_path).unlink(missing_ok=True)

    total_pages = len(pages)
    set_progress(
        request_id,
        "ocr",
        "OCR準備完了",
        0,
        total_pages,
    )

    results: list[dict[str, Any]] = []
    try:
        for index, page in enumerate(pages, start=1):
            if is_cancel_requested(request_id):
                set_progress(
                    request_id,
                    "canceled",
                    "中断しました",
                    max(0, index - 1),
                    total_pages,
                )
                clear_cancel_request(request_id)
                return {
                    "request_id": request_id,
                    "device": actual_device,
                    "task": normalized_task,
                    "linebreak_mode": normalized_linebreak_mode,
                    "state": "canceled",
                    "page_count": len(pages),
                    "results": results,
                }
            set_progress(
                request_id,
                "ocr",
                f"{index}/{total_pages}ページをOCR中",
                index,
                total_pages,
            )
            page_path = save_temp_png(page)
            try:
                async with GENERATE_SEMAPHORE:
                    raw_text, clean_text = await asyncio.to_thread(
                        glm_infer,
                        processor,
                        model,
                        str(page_path),
                        prompt,
                        max_new_tokens,
                        temperature,
                        request_id,
                    )
            finally:
                page_path.unlink(missing_ok=True)

            if is_cancel_requested(request_id):
                set_progress(
                    request_id,
                    "canceled",
                    "中断しました",
                    max(0, index - 1),
                    total_pages,
                )
                clear_cancel_request(request_id)
                return {
                    "request_id": request_id,
                    "device": actual_device,
                    "task": normalized_task,
                    "linebreak_mode": normalized_linebreak_mode,
                    "state": "canceled",
                    "page_count": len(pages),
                    "results": results,
                }

            item: dict[str, Any] = {
                "page": index,
                "text": (
                    normalize_linebreaks(clean_text, normalized_linebreak_mode)
                    if normalized_task != "extract_json"
                    else clean_text
                ),
                "raw": raw_text,
                "json": None,
            }
            if normalized_task == "extract_json":
                try:
                    item["json"] = json.loads(clean_text)
                except json.JSONDecodeError as exc:
                    item["error"] = f"JSON parse failed: {exc.msg}"
            results.append(item)
    except HTTPException:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", "APIエラー", 0, total_pages)
        raise
    except Exception as exc:
        clear_cancel_request(request_id)
        logger.exception("Inference failed")
        set_progress(request_id, "error", f"推論エラー: {exc}", 0, total_pages)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    set_progress(request_id, "done", "完了", total_pages, total_pages)
    clear_cancel_request(request_id)

    return {
        "request_id": request_id,
        "device": actual_device,
        "task": normalized_task,
        "linebreak_mode": normalized_linebreak_mode,
        "state": "done",
        "page_count": len(pages),
        "results": results,
    }
