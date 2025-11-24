import json
import os
import uuid
import asyncio
from glob import glob
import tempfile
import shutil
import re
import imghdr
import uvicorn
import base64
from base64 import b64encode
import zipfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
from functools import lru_cache
from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from loguru import logger
from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.utils.enum_class import MakeMode
from mineru.version import __version__
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

# Global configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_EXTENSIONS = pdf_suffixes + image_suffixes
CLEANUP_DELAY = 60  # 1 minute delay for cleanup (reduced to prevent resource buildup)

# Constants for better readability
REQUEST_ID_LENGTH = 8
TEMP_DIR_PREFIX = "mineru_"
OUTPUT_DIR_NAME = "output"
IMAGE_DIR_NAME = "images"

# Runtime mode configuration - only one mode can be active at a time
RUNTIME_MODE = os.environ.get("MINERU_MODE", "pipeline").lower()  # pipeline or vlm
SERVER_URL = os.environ.get("MINERU_SERVER_URL", None)  # For VLM client mode
VLM_BACKEND = os.environ.get("MINERU_VLM_BACKEND", "vlm-vllm-async-engine")  # Default VLM backend

# Backend configurations
PIPELINE_BACKENDS = ["pipeline"]
# VLM backends should use the "vlm-" prefix format that aio_do_parse expects
VLM_BACKENDS = ["vlm-vllm-async-engine"]

# Determine active backends based on runtime mode
if RUNTIME_MODE == "pipeline":
    ACTIVE_BACKENDS = PIPELINE_BACKENDS
    DEFAULT_BACKEND = "pipeline"
    logger.info("Running in PIPELINE mode")
elif RUNTIME_MODE == "vlm":
    ACTIVE_BACKENDS = [VLM_BACKEND] if VLM_BACKEND in VLM_BACKENDS else ["vlm-vllm-async-engine"]
    DEFAULT_BACKEND = ACTIVE_BACKENDS[0]
    logger.info(f"Running in VLM mode with backend: {DEFAULT_BACKEND}")
else:
    # Fallback to pipeline mode
    RUNTIME_MODE = "pipeline"
    ACTIVE_BACKENDS = PIPELINE_BACKENDS
    DEFAULT_BACKEND = "pipeline"
    logger.warning(f"Unknown mode '{RUNTIME_MODE}', falling back to pipeline mode")

ALL_BACKENDS = PIPELINE_BACKENDS + VLM_BACKENDS  # Keep for reference

# Backend-specific settings
BACKEND_CONFIGS = {
    "pipeline": {
        "supports_parse_method": True,
        "supports_formula_table": True,
        "default_parse_method": "pipeline",
        "result_subdir": lambda method: method,
        "model_output_suffix": "_model.json",
        "supports_ocr_lang": True,
    },

    "vlm-vllm-async-engine": {
        "supports_parse_method": True,
        "supports_formula_table": True,
        "default_parse_method": "vlm",
        "result_subdir": lambda method: "vlm",
        "model_output_suffix": "_model.json",
        "supports_ocr_lang": True,
        "requires_gpu": True,
        "supports_server_params": True,        
    }
}

# Cache commonly used values for better performance
SUPPORTED_EXTENSIONS_SET = frozenset(SUPPORTED_EXTENSIONS)
PIPELINE_BACKENDS_SET = frozenset(PIPELINE_BACKENDS)
VLM_BACKENDS_SET = frozenset(VLM_BACKENDS)

# Global connection tracking
active_connections = set()
connection_lock = asyncio.Lock()

# Create FastAPI app with proper lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting MinerU FastAPI server")

    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down MinerU FastAPI server")

        # Cancel cleanup task
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        # Force close any remaining connections
        async with connection_lock:
            for conn in active_connections.copy():
                try:
                    await conn.close()
                except:
                    pass
            active_connections.clear()

        # Force garbage collection
        import gc
        gc.collect()

async def periodic_cleanup():
    """Periodic cleanup of resources and connections with improved efficiency"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes

            # Force garbage collection
            import gc
            collected = gc.collect()

            # Log connection status more efficiently
            async with connection_lock:
                conn_count = len(active_connections)
            logger.info(f"Periodic cleanup: {collected} objects collected, {conn_count} active connections")

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {str(e)}")
            # Continue running even if there's an error

app = FastAPI(
    title="MinerU API",
    description="Document parsing API using MinerU",
    version=__version__,
    lifespan=lifespan
)

# Connection tracking middleware
@app.middleware("http")
async def connection_tracking_middleware(request, call_next):
    """Track and manage HTTP connections with improved efficiency"""
    connection_id = uuid.uuid4().hex[:REQUEST_ID_LENGTH]

    try:
        # Add connection to tracking
        async with connection_lock:
            active_connections.add(connection_id)

        # Process request
        response = await call_next(request)

        # Ensure response is properly closed to prevent CLOSE_WAIT
        if hasattr(response, 'headers'):
            response.headers["Connection"] = "close"
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"

        return response

    except Exception as e:
        logger.error(f"Connection {connection_id} error: {str(e)}")
        raise
    finally:
        # Always remove connection from tracking
        try:
            async with connection_lock:
                active_connections.discard(connection_id)
        except Exception as cleanup_error:
            logger.error(f"Error removing connection {connection_id}: {cleanup_error}")

# Add middleware for better performance and security
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Check if temp directory can be created
        test_dir = tempfile.mkdtemp(prefix="health_check_")
        safe_cleanup_directory(test_dir)

        health_info = {
            "status": "healthy",
            "version": __version__,
            "runtime_mode": RUNTIME_MODE,
            "active_backends": ACTIVE_BACKENDS,
            "default_backend": DEFAULT_BACKEND,
            "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
            "supported_extensions": SUPPORTED_EXTENSIONS
        }

        # Add VLM-specific health info
        if RUNTIME_MODE == "vlm":
            health_info["vlm_backend"] = DEFAULT_BACKEND
            if SERVER_URL:
                health_info["server_url"] = SERVER_URL
                health_info["server_configured"] = True
            else:
                health_info["server_configured"] = False


        return health_info

    except Exception as e:
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": __version__,
                "runtime_mode": RUNTIME_MODE
            },
            status_code=503
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MinerU Document Parsing API - Unified Interface",
        "version": __version__,
        "runtime_mode": RUNTIME_MODE,
        "active_backends": ACTIVE_BACKENDS,
        "default_backend": DEFAULT_BACKEND,
        "parse_endpoint": "/v2/parse/file",
        "docs": "/docs",
        "health": "/health",
        "features": {
            "unified_interface": True,
            "no_concurrency_limits": True,
            "supports_all_backends": True,
            "supports_all_file_types": SUPPORTED_EXTENSIONS
        },
        "mode_info": {
            "current_mode": RUNTIME_MODE,
            "description": "Pipeline OCR-based processing" if RUNTIME_MODE == "pipeline" else f"VLM-based processing using {DEFAULT_BACKEND}",
            "available_backends": ACTIVE_BACKENDS,
            "environment_variables": {
                "MINERU_MODE": "Set to 'pipeline' or 'vlm' to choose processing mode",
                "MINERU_VLM_BACKEND": "Choose VLM backend (must use vlm- prefix: vlm-vllm-async-engine)",
                "MINERU_SERVER_URL": "Server URL for VLM client mode (e.g., http://127.0.0.1:30000)"
            }
        }
    }

@app.get("/backends")
async def get_backends():
    """Get detailed backend information"""
    return {
        "runtime_mode": RUNTIME_MODE,
        "active_backends": ACTIVE_BACKENDS,
        "default_backend": DEFAULT_BACKEND,
        "all_backends": ALL_BACKENDS,
        "pipeline_backends": PIPELINE_BACKENDS,
        "vlm_backends": VLM_BACKENDS,
        "backend_configs": {k: v for k, v in BACKEND_CONFIGS.items() if k in ACTIVE_BACKENDS},
        "inactive_backends": [b for b in ALL_BACKENDS if b not in ACTIVE_BACKENDS],
        "server_url": SERVER_URL if RUNTIME_MODE == "vlm" else None
    }

@app.get("/backends/{backend}/config")
async def get_backend_config(backend: str):
    """Get configuration details for a specific backend"""
    if backend not in ALL_BACKENDS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Backend '{backend}' not found. All backends: {ALL_BACKENDS}"}
        )

    if backend not in ACTIVE_BACKENDS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Backend '{backend}' not available in {RUNTIME_MODE} mode",
                "active_backends": ACTIVE_BACKENDS,
                "runtime_mode": RUNTIME_MODE,
                "note": f"To use '{backend}', restart server with appropriate MINERU_MODE environment variable"
            }
        )

    config = BACKEND_CONFIGS[backend]

    # Build parameter recommendations
    recommended_params = {
        "required": [],
        "optional": [],
        "not_supported": []
    }

    # Common parameters
    common_params = ["start_page_id", "end_page_id", "tp_size", "dp_size", "enable_torch_compile"]
    recommended_params["optional"].extend(common_params)

    # Backend-specific parameters
    if config.get("requires_server_url"):
        recommended_params["required"].append("server_url")
    elif "server_url" not in recommended_params["required"]:
        recommended_params["optional"].append("server_url")

    if config.get("supports_parse_method"):
        recommended_params["optional"].append("parse_method")
        recommended_params["parse_methods"] = ["auto", "txt", "ocr"]
    else:
        recommended_params["not_supported"].append("parse_method")
        recommended_params["fixed_parse_method"] = config["default_parse_method"]

    if config.get("supports_formula_table"):
        recommended_params["optional"].extend(["formula_enable", "table_enable"])
    else:
        recommended_params["not_supported"].extend(["formula_enable", "table_enable"])

    if config.get("supports_ocr_lang"):
        recommended_params["optional"].append("lang")
        recommended_params["supported_languages"] = ["ch", "en", "korean", "japan", "chinese_cht", "ta", "te", "ka"]
    else:
        recommended_params["not_supported"].append("lang")

    if config.get("supports_server_params"):
        vlm_params = ["temperature", "top_p", "top_k", "repetition_penalty",
                      "presence_penalty", "no_repeat_ngram_size", "max_new_tokens"]
        recommended_params["optional"].extend(vlm_params)
    else:
        vlm_params = ["temperature", "top_p", "top_k", "repetition_penalty",
                      "presence_penalty", "no_repeat_ngram_size", "max_new_tokens"]
        recommended_params["not_supported"].extend(vlm_params)

    return {
        "backend": backend,
        "config": config,
        "parameters": recommended_params,
        "example_usage": {
            "basic": f"POST /v2/parse/file with backend={backend}",
            "with_params": get_example_params(backend, config)
        }
    }

def get_example_params(backend: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate example parameters for a backend"""
    example = {"backend": backend}

    if config.get("supports_parse_method"):
        example["parse_method"] = "auto"

    if config.get("supports_formula_table"):
        example["formula_enable"] = True
        example["table_enable"] = True

    if config.get("supports_ocr_lang"):
        example["lang"] = "ch"

    if config.get("requires_server_url"):
        example["server_url"] = "http://127.0.0.1:30000"

    if config.get("supports_server_params"):
        example.update({
            "temperature": 0.0001,
            "top_p": 0.8,
            "max_new_tokens": 8192
        })

    return example

def encode_image(image_path: str) -> Optional[str]:
    """Encode image using base64 with error handling"""
    try:
        with open(image_path, "rb") as f:
            return b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {str(e)}")
        return None

def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """Read inference result from result file with error handling"""
    try:
        result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
        logger.debug(f"Trying to read: {result_file_path}")
        logger.debug(f"File exists: {os.path.exists(result_file_path)}")
        
        if os.path.exists(result_file_path):
            with open(result_file_path, "r", encoding="utf-8") as fp:
                content = fp.read()
                logger.debug(f"Successfully read {len(content)} bytes from {file_suffix_identifier}")
                return content
        else:
            logger.warning(f"File not found: {result_file_path}")
    except Exception as e:
        logger.error(f"Failed to read result file {result_file_path}: {str(e)}")
    return None

async def delayed_cleanup(directory_path: str, delay: int = CLEANUP_DELAY):
    """Delayed cleanup of directory to allow for any pending operations"""
    try:
        await asyncio.sleep(delay)
        safe_cleanup_directory(directory_path)
    except Exception as e:
        logger.error(f"Error in delayed cleanup: {str(e)}")
    finally:
        # Force garbage collection to free memory
        import gc
        gc.collect()


async def immediate_cleanup(directory_path: str):
    """Immediate cleanup for error cases"""
    try:
        if safe_cleanup_directory(directory_path):
            logger.debug(f"Immediate cleanup successful for {directory_path}")
        else:
            logger.warning(f"Immediate cleanup failed for {directory_path}")
    except Exception as e:
        logger.error(f"Error in immediate cleanup: {str(e)}")
    finally:
        import gc
        gc.collect()


def safe_cleanup_directory(directory_path: str) -> bool:
    """Safely clean up directory with proper error handling and improved efficiency"""
    if not os.path.exists(directory_path):
        return True

    try:
        logger.debug(f"Cleaning up directory: {directory_path}")
        # Try to remove the entire directory first
        shutil.rmtree(directory_path, ignore_errors=False)
        logger.debug("Directory cleaned up successfully")
        return True
    except PermissionError as e:
        logger.warning(f"Permission error cleaning up directory: {str(e)}")
        # Try to fix permissions and retry
        try:
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for name in files + dirs:  # Handle both files and directories
                    try:
                        path = os.path.join(root, name)
                        os.chmod(path, 0o777)  # Ensure write permissions
                    except Exception:
                        pass  # Ignore errors when changing permissions
            # Final attempt to remove the directory
            shutil.rmtree(directory_path, ignore_errors=True)
            return True
        except Exception as perm_error:
            logger.error(f"Failed to cleanup directory after permission fix: {str(perm_error)}")
            return False
    except Exception as e:
        logger.error(f"Error cleaning up directory: {str(e)}")
        # Last resort: try to remove individual files with error ignoring
        shutil.rmtree(directory_path, ignore_errors=True)
        # Check if directory still exists
        if os.path.exists(directory_path):
            logger.error(f"Failed to completely remove directory: {directory_path}")
            return False
        return True

def validate_file_size(file_content: bytes) -> bool:
    """Validate file size efficiently"""
    # Use direct comparison without creating len() overhead for large files
    return len(file_content) <= MAX_FILE_SIZE


async def validate_file_size_async(file: UploadFile) -> bool:
    """
    异步验证文件大小，避免将整个文件加载到内存中
    对于大文件更高效
    """
    # 对于较小的文件，直接读取
    if file.size and file.size <= MAX_FILE_SIZE:
        return True
    
    # 对于大小未知或较大的文件，需要读取检查
    if not file.size:
        # 读取文件检查大小
        content = await file.read()
        file.file.seek(0)  # 重置文件指针
        return len(content) <= MAX_FILE_SIZE
    
    return False


def get_file_info(file: UploadFile) -> Dict[str, Any]:
    """Extract file information with better error handling"""
    try:
        file_path = Path(file.filename or "unknown")
        return {
            "filename": file.filename or "unknown",
            "stem": file_path.stem,
            "suffix": file_path.suffix.lower() if file_path.suffix else "",
            "size": getattr(file, 'size', 0) or 0
        }
    except Exception as e:
        logger.warning(f"Error extracting file info: {e}")
        return {
            "filename": getattr(file, 'filename', "unknown") or "unknown",
            "stem": "unknown",
            "suffix": "",
            "size": getattr(file, 'size', 0) or 0
        }


def get_file_info_with_guess(temp_file_path: str) -> Dict[str, Any]:
    """使用guess_suffix_by_path提取文件信息，提供更准确的文件类型检测"""
    try:
        # 使用guess_suffix_by_path获取文件后缀
        file_suffix = guess_suffix_by_path(temp_file_path)
        file_path = Path(temp_file_path)
        
        return {
            "filename": file_path.name,
            "stem": file_path.stem,
            "suffix": f".{file_suffix}" if file_suffix else file_path.suffix.lower(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }
    except Exception as e:
        logger.warning(f"Error extracting file info with guess: {e}")
        # 回退到原始方法
        try:
            file_path = Path(temp_file_path)
            return {
                "filename": file_path.name,
                "stem": file_path.stem,
                "suffix": file_path.suffix.lower() if file_path.suffix else "",
                "size": file_path.stat().st_size if file_path.exists() else 0
            }
        except Exception as fallback_e:
            logger.warning(f"Fallback error extracting file info: {fallback_e}")
            return {
                "filename": "unknown",
                "stem": "unknown",
                "suffix": "",
                "size": 0
            }


def validate_backend_config(backend: str, parse_method: str, server_url: Optional[str]) -> Dict[str, Any]:
    """Validate backend configuration and return adjusted parameters"""
    # Check if backend is available in current runtime mode
    if backend not in ACTIVE_BACKENDS:
        if backend in ALL_BACKENDS:
            # Backend exists but not active in current mode
            raise ValueError(f"Backend '{backend}' not available in {RUNTIME_MODE} mode. Available backends: {ACTIVE_BACKENDS}")
        else:
            # Backend doesn't exist at all
            raise ValueError(f"Unknown backend: {backend}. Available backends: {ACTIVE_BACKENDS}")

    config = BACKEND_CONFIGS[backend]
    result = {"backend": backend, "config": config}

    # Adjust parse method based on backend
    if not config["supports_parse_method"]:
        result["parse_method"] = config["default_parse_method"]
        if parse_method != "auto" and parse_method != config["default_parse_method"]:
            result["warning"] = f"Parse method '{parse_method}' not supported by {backend}, using '{config['default_parse_method']}'"
    else:
        result["parse_method"] = parse_method

    # Check server URL requirement
    if config.get("requires_server_url"):
        # Use environment variable if not provided in request
        actual_server_url = server_url or SERVER_URL
        if not actual_server_url:
            raise ValueError(f"Backend '{backend}' requires server_url parameter or MINERU_SERVER_URL environment variable")
        result["server_url"] = actual_server_url

    return result

@lru_cache(maxsize=128)
def get_result_directory_cached(output_dir: str, file_name: str, backend: str, parse_method: str) -> str:
    """缓存版本的获取结果目录函数，提高重复请求的性能"""
    config = BACKEND_CONFIGS[backend]
    subdir = config["result_subdir"](parse_method)
    return os.path.join(output_dir, file_name, subdir)


def get_result_directory(output_dir: str, file_name: str, backend: str, parse_method: str) -> str:
    """Get the result directory path based on backend type with caching for better performance"""
    # 对于高并发场景，可以使用缓存版本
    # return get_result_directory_cached(output_dir, file_name, backend, parse_method)
    # 为了简单起见，我们直接返回计算结果
    config = BACKEND_CONFIGS[backend]
    subdir = config["result_subdir"](parse_method)
    return os.path.join(output_dir, file_name, subdir)


@lru_cache(maxsize=64)
def get_model_output_suffix_cached(backend: str) -> str:
    """缓存版本的获取模型输出后缀函数"""
    return BACKEND_CONFIGS[backend]["model_output_suffix"]


def get_model_output_suffix(backend: str) -> str:
    """Get the model output file suffix based on backend"""
    # 对于高并发场景，可以使用缓存版本
    # return get_model_output_suffix_cached(backend)
    # 为了简单起见，我们直接返回计算结果
    return BACKEND_CONFIGS[backend]["model_output_suffix"]


def prepare_backend_params(backend: str, **kwargs) -> Dict[str, Any]:
    """Prepare parameters specific to backend type with improved efficiency"""
    config = BACKEND_CONFIGS[backend]
    params = {}

    # Common parameters - use dictionary comprehension for better performance
    common_params = {
        "start_page_id", "end_page_id", "server_url"
    }
    
    # Filter and add common parameters that exist in kwargs
    params.update({k: v for k, v in kwargs.items() if k in common_params})

    # Backend-specific parameters
    if backend in VLM_BACKENDS_SET and config.get("supports_server_params"):
        # 仅转发引擎级参数到 AsyncEngineArgs，排除生成采样参数
        vlm_engine_params = {
            "tensor_parallel_size",
            "data_parallel_size",
            "gpu_memory_utilization",
            "port",
            "logits_processors",
            "enable_torch_compile",
        }
        # 过滤并添加允许的引擎参数
        params.update({k: v for k, v in kwargs.items() if k in vlm_engine_params})

    # Pipeline-specific parameters (excluding formula_enable and table_enable as they're passed separately)
    if backend in PIPELINE_BACKENDS:
        # These are handled separately in the aio_do_parse call
        pass

    return params


def get_mime_type(image_path):
    """Dynamically detect image MIME type with enhanced format support"""
    # 首先尝试使用imghdr
    img_type = imghdr.what(image_path)
    if img_type:
        return f"image/{img_type}"
    
    # Fallback: 基于文件扩展名
    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    mime_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp',
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        'jp2': 'image/jp2',
    }
    return mime_map.get(ext, 'application/octet-stream')


async def process_images_async(image_dir: str, backend: str = "pipeline", request_id: str = "unknown") -> Dict[str, str]:
    """
    增强的异步图片处理，支持更多格式和更好的错误处理
    
    Args:
        image_dir: 图像目录路径
        backend: 后端类型（用于调试日志）
        request_id: 请求ID用于日志追踪
        
    Returns:
        Dict[str, str]: 图像文件名到base64编码的映射
    """
    # 扩展支持的图片格式
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff', 'tif', 'jp2']
    image_patterns = [f"*.{ext}" for ext in image_extensions]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob(os.path.join(image_dir, pattern)))
    
    if not image_paths:
        logger.debug(f"[{request_id}] No images found in {image_dir}")
        return {}
    
    logger.info(f"[{request_id}] Found {len(image_paths)} images in {image_dir} for backend {backend}")
    
    images_data = {}
    failed_count = 0
    
    for image_path in image_paths:
        try:
            # 验证文件存在且可读
            if not os.path.exists(image_path):
                logger.warning(f"[{request_id}] Image file not found: {image_path}")
                failed_count += 1
                continue
            
            if not os.access(image_path, os.R_OK):
                logger.warning(f"[{request_id}] Image file not readable: {image_path}")
                failed_count += 1
                continue
            
            filename = os.path.basename(image_path)
            base64_str = encode_image(image_path)
            
            if base64_str:
                mime_type = get_mime_type(image_path)
                images_data[filename] = f"data:{mime_type};base64,{base64_str}"
                logger.debug(f"[{request_id}] Successfully processed image: {filename} ({mime_type})")
            else:
                logger.warning(f"[{request_id}] Failed to encode image: {filename}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"[{request_id}] Error processing image {image_path}: {str(e)}")
            failed_count += 1
            continue
    
    if failed_count > 0:
        logger.warning(f"[{request_id}] Failed to process {failed_count} out of {len(image_paths)} images")
    
    logger.info(f"[{request_id}] Successfully processed {len(images_data)} images")
    return images_data


def replace_markdown_images(markdown_content: str, images_data: Dict[str, str], 
                           request_id: str = "unknown") -> str:
    """
    增强的Markdown图片替换函数
    
    支持多种图片引用格式：
    - ![](images/xxx.jpg)
    - ![alt text](images/xxx.jpg)
    - ![](./images/xxx.jpg)
    - ![](_xxx_images/xxx.jpg)
    
    Args:
        markdown_content: 原始Markdown内容
        images_data: 图片文件名到base64 data URI的映射
        request_id: 请求ID用于日志
        
    Returns:
        str: 替换后的Markdown内容
    """
    if not markdown_content or not images_data:
        return markdown_content
    
    replaced_count = 0
    failed_refs = []
    
    def replace_image(match):
        nonlocal replaced_count
        
        # 提取alt文本和图片路径
        # match.group(1) 是 alt 文本, match.group(2) 是路径
        alt_text = match.group(1) if match.lastindex >= 1 else ""
        image_path = match.group(2) if match.lastindex >= 2 else match.group(1)
        
        # 提取文件名（支持不同路径格式）
        # 处理路径如: images/xxx.jpg, ./images/xxx.jpg, _stem_images/xxx.jpg
        filename = image_path.split('/')[-1]
        
        if filename in images_data:
            replaced_count += 1
            # 保留原有的alt文本，如果没有则使用文件名
            display_alt = alt_text if alt_text else filename
            return f'![{display_alt}]({images_data[filename]})'
        else:
            # 记录未找到的图片引用
            if filename not in failed_refs:
                failed_refs.append(filename)
            logger.debug(f"[{request_id}] Image not found in data: {filename}")
            return match.group(0)  # 保持原样
    
    # 支持多种Markdown图片格式
    # Pattern 1: ![alt text](path) or ![](path) - 标准格式
    # Pattern 2: 相对路径 ./images/
    # Pattern 3: 特定命名模式 _xxx_images/
    patterns = [
        r'!\[([^\]]*)\]\(([^)]*?/?images/[^)]+)\)',  # 带或不带alt文本的标准路径
        r'!\[([^\]]*)\]\((\./[^)]*?/?images/[^)]+)\)',  # 相对路径
        r'!\[([^\]]*)\]\(([^)]*?_images/[^)]+)\)',  # 特定命名模式
    ]
    
    result = markdown_content
    for pattern in patterns:
        result = re.sub(pattern, replace_image, result)
    
    if replaced_count > 0:
        logger.info(f"[{request_id}] Replaced {replaced_count} image references in markdown")
    
    if failed_refs:
        unique_failed = list(set(failed_refs))
        logger.warning(f"[{request_id}] Could not find {len(unique_failed)} images: {unique_failed[:5]}...")
    
    return result


def validate_and_fix_images(response_data: dict, request_id: str = "unknown") -> dict:
    """
    验证并诊断图片路径问题
    
    检查:
    1. markdown中的图片引用是否有对应的images数据
    2. 空路径图片的处理
    3. 缺失图片的详细日志记录
    
    Args:
        response_data: 响应数据字典
        request_id: 请求ID用于日志
        
    Returns:
        dict: 更新后的响应数据（添加diagnostics信息）
    """
    if "markdown" not in response_data:
        return response_data
    
    markdown = response_data.get("markdown", "")
    images_data = response_data.get("images", {})
    
    # 统计信息
    stats = {
        "total_refs": 0,
        "empty_refs": 0,
        "missing_refs": 0,
        "valid_refs": 0,
        "data_uri_refs": 0,
    }
    
    # 检测空路径图片引用 ![xxx]() 或 ![]()
    empty_pattern = r'!\[([^\]]*)\]\(\s*\)'
    empty_refs = re.findall(empty_pattern, markdown)
    stats["empty_refs"] = len(empty_refs)
    
    if empty_refs:
        logger.warning(
            f"[{request_id}] Found {len(empty_refs)} empty image references - "
            "likely due to image_path generation failure in VLM backend"
        )
    
    # 检测所有图片引用
    all_refs_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    all_refs = re.findall(all_refs_pattern, markdown)
    stats["total_refs"] = len(all_refs)
    
    missing_images = []
    for alt, path in all_refs:
        # 跳过已经是data URI的引用
        if path.startswith('data:'):
            stats["data_uri_refs"] += 1
            stats["valid_refs"] += 1
            continue
        
        # 检查文件路径引用
        if 'images/' in path or '_images/' in path:
            filename = path.split('/')[-1]
            if filename in images_data:
                stats["valid_refs"] += 1
            else:
                stats["missing_refs"] += 1
                missing_images.append({
                    "filename": filename,
                    "alt": alt,
                    "path": path
                })
    
    # 记录缺失的图片
    if missing_images:
        logger.warning(
            f"[{request_id}] Found {len(missing_images)} referenced images not in data: "
            f"{[img['filename'] for img in missing_images[:5]]}"
        )
    
    # 添加诊断信息到响应
    if "diagnostics" not in response_data:
        response_data["diagnostics"] = {}
    
    response_data["diagnostics"]["images"] = {
        "total_references": stats["total_refs"],
        "empty_references": stats["empty_refs"],
        "missing_references": stats["missing_refs"],
        "valid_references": stats["valid_refs"],
        "data_uri_references": stats["data_uri_refs"],
        "total_image_files": len(images_data),
        "missing_details": missing_images[:10] if missing_images else [],
    }
    
    # 记录汇总信息
    logger.info(
        f"[{request_id}] Image diagnostics: "
        f"refs={stats['total_refs']}, valid={stats['valid_refs']}, "
        f"empty={stats['empty_refs']}, missing={stats['missing_refs']}, "
        f"files={len(images_data)}"
    )
    
    return response_data


def validate_and_enhance_table_content(content_list_str: str, request_id: str = "unknown") -> tuple:
    """
    验证和诊断表格内容
    
    处理:
    1. 检查表格HTML的有效性
    2. 记录缺失表格内容的详细信息
    3. 提供表格内容质量统计
    
    Args:
        content_list_str: content_list JSON字符串
        request_id: 请求ID用于日志
        
    Returns:
        tuple: (content_list_str, diagnostics_dict)
    """
    try:
        content_list = json.loads(content_list_str)
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] Failed to parse content_list JSON: {e}")
        return content_list_str, {}
    
    if not isinstance(content_list, list):
        return content_list_str, {}
    
    # 统计信息
    stats = {
        "total_tables": 0,
        "tables_with_html": 0,
        "tables_with_image": 0,
        "tables_empty": 0,
        "tables_with_both": 0,
    }
    
    empty_tables = []
    invalid_html_tables = []
    
    for idx, item in enumerate(content_list):
        if item.get('type') != 'table':
            continue
        
        stats["total_tables"] += 1
        page_idx = item.get('page_idx', 'unknown')
        
        has_html = bool(item.get('table_body'))
        has_image = bool(item.get('img_path'))
        
        if has_html and has_image:
            stats["tables_with_both"] += 1
            stats["tables_with_html"] += 1
        elif has_html:
            stats["tables_with_html"] += 1
            
            # 验证HTML基本结构
            html_content = item['table_body']
            if not html_content.strip():
                invalid_html_tables.append({
                    "index": idx,
                    "page": page_idx,
                    "issue": "empty_html"
                })
            elif '<table' not in html_content.lower():
                invalid_html_tables.append({
                    "index": idx,
                    "page": page_idx,
                    "issue": "missing_table_tag"
                })
                logger.warning(
                    f"[{request_id}] Invalid table HTML at index {idx}, page {page_idx}: "
                    "missing <table> tag"
                )
        elif has_image:
            stats["tables_with_image"] += 1
        else:
            stats["tables_empty"] += 1
            empty_tables.append({
                "index": idx,
                "page": page_idx,
                "bbox": item.get('bbox', [])
            })
            logger.warning(
                f"[{request_id}] Empty table found at index {idx}, page {page_idx} - "
                "no HTML content and no image path"
            )
    
    # 构建诊断信息
    diagnostics = {
        "total_tables": stats["total_tables"],
        "with_html": stats["tables_with_html"],
        "with_image_only": stats["tables_with_image"],
        "with_both": stats["tables_with_both"],
        "empty": stats["tables_empty"],
        "invalid_html": len(invalid_html_tables),
        "empty_table_details": empty_tables[:10] if empty_tables else [],
        "invalid_html_details": invalid_html_tables[:10] if invalid_html_tables else [],
    }
    
    # 记录汇总信息
    if stats["total_tables"] > 0:
        logger.info(
            f"[{request_id}] Table diagnostics: "
            f"total={stats['total_tables']}, html={stats['tables_with_html']}, "
            f"image_only={stats['tables_with_image']}, empty={stats['tables_empty']}, "
            f"invalid_html={len(invalid_html_tables)}"
        )
        
        # 警告：如果有大量空表格
        if stats["tables_empty"] > 0:
            empty_ratio = stats["tables_empty"] / stats["total_tables"]
            if empty_ratio > 0.3:
                logger.warning(
                    f"[{request_id}] High empty table ratio: {empty_ratio:.1%} "
                    f"({stats['tables_empty']}/{stats['total_tables']}) - "
                    "possible VLM backend issue"
                )
    
    return content_list_str, diagnostics


def is_invoice_table(html_content: str) -> bool:
    """
    检测是否是发票表格
    
    发票表格特征:
    - 包含"税率"或"税额"
    - 包含"金额"
    - 通常有"合计"行
    """
    if not html_content:
        return False
    
    content_lower = html_content.lower()
    invoice_keywords = ['税率', '税额', '金额', '单价', '数量']
    
    # 至少包含2个发票关键词
    keyword_count = sum(1 for keyword in invoice_keywords if keyword in html_content)
    return keyword_count >= 2


def fix_invoice_table_html(html_content: str, request_id: str = "unknown") -> str:
    """
    修复发票表格HTML的常见问题
    
    修复内容:
    1. 标准化表格列数（发票通常是7-8列）
    2. 修复"合计"行的格式
    3. 规范化数字精度
    4. 修复列对齐问题
    
    Args:
        html_content: 原始HTML内容
        request_id: 请求ID用于日志
        
    Returns:
        str: 修复后的HTML内容
    """
    if not html_content or not is_invoice_table(html_content):
        return html_content
    
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            logger.warning(f"[{request_id}] Invoice table HTML missing <table> tag")
            return html_content
        
        rows = table.find_all('tr')
        if len(rows) < 2:
            return html_content
        
        # 统计最常见的列数
        col_counts = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            col_counts.append(len(cells))
        
        if not col_counts:
            return html_content
        
        # 发票标准列数通常是7-8列
        expected_cols = max(set(col_counts), key=col_counts.count)
        
        fixed_rows = []
        for idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            current_cols = len(cells)
            
            # 如果是合计行，特殊处理
            row_text = row.get_text(strip=True)
            if '合计' in row_text or '小计' in row_text:
                # 合计行通常前几列合并，后面是金额和税额
                if current_cols < expected_cols:
                    # 补充空单元格
                    while len(cells) < expected_cols - 2:
                        new_cell = soup.new_tag('td')
                        cells.insert(0, new_cell)
                    
                    # 重建合计行
                    new_row = soup.new_tag('tr')
                    for cell in cells:
                        new_row.append(cell)
                    fixed_rows.append(new_row)
                    logger.debug(f"[{request_id}] Fixed invoice total row: {current_cols} -> {len(cells)} cols")
                    continue
            
            # 规范化数字精度
            for cell in cells:
                cell_text = cell.get_text(strip=True)
                # 检测异常精度的数字（如176.991150442478）
                if cell_text and '.' in cell_text:
                    try:
                        num = float(cell_text)
                        # 如果小数位超过4位，规范化为2位
                        if len(cell_text.split('.')[-1]) > 4:
                            cell.string = f"{num:.2f}"
                            logger.debug(f"[{request_id}] Normalized number precision: {cell_text} -> {num:.2f}")
                    except ValueError:
                        pass
            
            fixed_rows.append(row)
        
        # 重建表格
        new_table = soup.new_tag('table')
        for row in fixed_rows:
            new_table.append(row)
        
        # 保留原有的table属性
        if table.attrs:
            new_table.attrs.update(table.attrs)
        
        result = str(new_table)
        
        if result != html_content:
            logger.info(f"[{request_id}] Fixed invoice table HTML structure")
        
        return result
        
    except ImportError:
        # 无 bs4 时，使用正则对标签文本中的长小数进行回退修复
        import re

        def _normalize_decimals_in_text(text: str) -> str:
            def _fmt(m):
                try:
                    return f"{float(m.group(0)):.2f}"
                except Exception:
                    return m.group(0)
            # 仅规范化小数位超过4位的数字
            return re.sub(r"\b\d+\.\d{5,}\b", _fmt, text)

        try:
            new_html = re.sub(r"(?<=>)([^<>]*?)(?=<)", lambda m: _normalize_decimals_in_text(m.group(1)), html_content)
            if new_html != html_content:
                logger.info(f"[{request_id}] Fallback normalized invoice decimals without BeautifulSoup")
            return new_html
        except Exception:
            return html_content
    except Exception as e:
        logger.error(f"[{request_id}] Error fixing invoice table HTML: {e}")
        return html_content


def optimize_vlm_params_for_invoice(request_id: str = "unknown") -> dict:
    """
    为发票解析优化VLM参数
    
    优化策略:
    1. 强制启用table_enable
    2. 禁用formula_enable（发票不需要公式）
    3. 建议使用特定的parse_method
    
    Args:
        request_id: 请求ID用于日志
        
    Returns:
        dict: 优化建议
    """
    recommendations = {
        "table_enable": True,
        "formula_enable": False,
        "parse_method": "auto",
        "backend_preference": ["pipeline", "vlm-vllm-async-engine"],
        "reason": "Invoice documents benefit from table recognition and OCR accuracy"
    }
    
    # 设置环境变量
    # os.environ['MINERU_VLM_TABLE_ENABLE'] = 'true'
    # os.environ['MINERU_VLM_FORMULA_ENABLE'] = 'false'
    
    logger.info(
        f"[{request_id}] Optimized VLM parameters for invoice: "
        f"table_enable=true, formula_enable=false"
    )
    
    return recommendations


def enhance_content_list_for_invoice(content_list_str: str, request_id: str = "unknown") -> str:
    """
    增强发票的content_list，修复表格HTML
    
    Args:
        content_list_str: content_list JSON字符串
        request_id: 请求ID用于日志
        
    Returns:
        str: 增强后的content_list JSON字符串
    """
    try:
        content_list = json.loads(content_list_str)
    except json.JSONDecodeError:
        return content_list_str
    
    if not isinstance(content_list, list):
        return content_list_str
    
    modified = False
    
    for item in content_list:
        if item.get('type') == 'table' and item.get('table_body'):
            original_html = item['table_body']
            
            # 检测并修复发票表格
            if is_invoice_table(original_html):
                fixed_html = fix_invoice_table_html(original_html, request_id)
                
                if fixed_html != original_html:
                    item['table_body'] = fixed_html
                    modified = True
    
    if modified:
        logger.info(f"[{request_id}] Enhanced invoice tables in content_list")
        return json.dumps(content_list, ensure_ascii=False)
    
    return content_list_str


def extract_text_from_block(block: dict) -> list:
    """
    从block中提取所有文本内容
    
    支持多种block结构:
    - 简单block: 直接包含lines
    - 嵌套block: 包含blocks数组（如list, table等）
    
    Args:
        block: block字典
        
    Returns:
        list: 提取的文本列表
    """
    texts = []
    
    # 处理直接包含lines的block
    if "lines" in block:
        for line in block["lines"]:
            if "spans" in line:
                for span in line["spans"]:
                    if "content" in span:
                        texts.append(span["content"])
    
    # 处理嵌套结构（如list, table等）
    elif "blocks" in block:
        for sub_block in block["blocks"]:
            texts.extend(extract_text_from_block(sub_block))
    
    return texts


def extract_invoice_metadata_from_all_blocks(middle_json_str: str, request_id: str = "unknown") -> dict:
    """
    从所有blocks中提取发票元数据（不限于discarded_blocks）
    
    策略:
    - 检查discarded_blocks（VLM模型正确识别为header的情况）
    - 检查para_blocks的前5个（VLM模型将header识别为text/title的情况）
    - 使用灵活的模式匹配识别发票号码和日期
    
    Args:
        middle_json_str: middle_json JSON字符串
        request_id: 请求ID用于日志
        
    Returns:
        dict: 提取的元数据
    """
    try:
        middle_json = json.loads(middle_json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] Failed to parse middle_json: {e}")
        return {}
    
    metadata = {
        "numbers": [],
        "dates": [],
        "stamps": [],
        "headers": [],
        "key_value_pairs": {},
        "flags": {
            "is_electronic_invoice": False
        },
        "issuer_names": []
    }
    
    # 只处理第一页
    pdf_info = middle_json.get("pdf_info", [])
    if not pdf_info:
        return metadata
    
    page_info = pdf_info[0]
    discarded = page_info.get("discarded_blocks", [])
    para_blocks = page_info.get("para_blocks", [])
    
    # 合并discarded_blocks和para_blocks的前5个
    # 前5个para_blocks通常包含文档头部信息
    blocks_to_check = discarded + para_blocks[:5]
    
    logger.info(f"[{request_id}] Checking {len(discarded)} discarded blocks + {min(5, len(para_blocks))} para blocks for metadata")
    
    for block in blocks_to_check:
        block_type = block.get("type", "unknown")
        
        # 提取文本内容
        texts = extract_text_from_block(block)
        full_text = " ".join(texts)
        
        if not full_text.strip():
            continue
        
        # 1. 搜索长数字串（可能是发票号码、订单号等）
        numbers = re.findall(r'\b\d{10,30}\b', full_text)
        for num in numbers:
            if num not in [n["value"] for n in metadata["numbers"]]:
                metadata["numbers"].append({
                    "value": num,
                    "length": len(num),
                    "block_type": block_type,
                    "context": full_text[:100]
                })
        
        # 2. 搜索日期（多种格式）
        # 格式1: 2024年12月10日
        dates_cn = re.findall(r'(\d{4})[年](\d{1,2})[月](\d{1,2})[日]?', full_text)
        for y, m, d in dates_cn:
            date_str = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            if date_str not in [d["value"] for d in metadata["dates"]]:
                metadata["dates"].append({
                    "value": date_str,
                    "format": "YYYY年MM月DD日",
                    "block_type": block_type,
                    "context": full_text[:100]
                })
        
        # 格式2: 2024-12-10 或 2024/12/10
        dates_slash = re.findall(r'(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})', full_text)
        for y, m, d in dates_slash:
            date_str = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            if date_str not in [d["value"] for d in metadata["dates"]]:
                metadata["dates"].append({
                    "value": date_str,
                    "format": "YYYY-MM-DD",
                    "block_type": block_type,
                    "context": full_text[:100]
                })
        
        # 3. 识别印章相关文字
        if re.search(r'(印章|签章|盖章|专用章|财务章|发票专用章)', full_text):
            metadata["stamps"].append({
                "text": full_text[:100],
                "block_type": block_type
            })
        
        # 4. 识别header文本
        if block_type in ["header", "title"]:
            metadata["headers"].append(full_text[:200])
        
        # 5. 识别电子发票标识
        if re.search(r'电子发票', full_text):
            metadata["flags"]["is_electronic_invoice"] = True
        
        # 6. 识别开具机构
        issuer_match = re.findall(r'(税务局|税务管理局|国家税务总局|[^，。]{2,10}税务局)', full_text)
        for issuer in issuer_match:
            if issuer not in metadata["issuer_names"]:
                metadata["issuer_names"].append(issuer)
        
        # 7. 尝试提取键值对
        kv_pattern = r'([^:：\n]{2,10})[：:]\s*([^\n]{1,50})'
        kv_matches = re.findall(kv_pattern, full_text)
        for key, value in kv_matches:
            key = key.strip()
            value = value.strip()
            if len(key) >= 2 and len(value) >= 1:
                if key not in metadata["key_value_pairs"]:
                    metadata["key_value_pairs"][key] = value
    
    # 推断主要字段
    metadata["inferred"] = _infer_primary_fields(metadata, request_id)
    
    # 记录提取结果
    logger.info(
        f"[{request_id}] Extracted from all blocks: "
        f"numbers={len(metadata['numbers'])}, "
        f"dates={len(metadata['dates'])}, "
        f"kv_pairs={len(metadata['key_value_pairs'])}"
    )
    
    return metadata


def extract_discarded_invoice_metadata(middle_json_str: str, request_id: str = "unknown") -> dict:
    """
    从middle_json的discarded_blocks中智能提取发票元数据
    
    使用灵活的模式匹配和启发式规则，而非硬编码：
    - 自适应识别各种长度的数字串（发票号、订单号等）
    - 支持多种日期格式的自动检测
    - 智能识别印章、签名等关键元素
    
    Args:
        middle_json_str: middle_json JSON字符串
        request_id: 请求ID用于日志
        
    Returns:
        dict: 提取的元数据，包含所有识别到的字段
    """
    try:
        middle_json = json.loads(middle_json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[{request_id}] Failed to parse middle_json: {e}")
        return {}
    
    metadata = {
        "numbers": [],          # 所有识别到的长数字串
        "dates": [],            # 所有识别到的日期
        "stamps": [],           # 印章相关文字
        "headers": [],          # 所有header文本
        "key_value_pairs": {},  # 识别到的键值对
        "flags": {              # 额外标志位
            "is_electronic_invoice": False
        },
        "issuer_names": []      # 可能的开具机构（税务局等）
    }
    
    pdf_info = middle_json.get("pdf_info", [])
    
    for page_info in pdf_info:
        discarded_blocks = page_info.get("discarded_blocks", [])
        
        if not discarded_blocks:
            continue
        
        logger.debug(f"[{request_id}] Processing {len(discarded_blocks)} discarded blocks")
        
        for block in discarded_blocks:
            # 提取文本内容
            text_content = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") == "text":
                        text_content += span.get("content", "") + " "
            
            text_content = text_content.strip()
            if not text_content:
                continue
            
            # 1. 提取所有长数字串（可能是发票号、订单号、流水号等）
            # 匹配10位以上的连续数字
            # 发票相关编号可能为 8-30 位，这里更宽松一些
            number_pattern = r'\b\d{8,30}\b'
            for match in re.finditer(number_pattern, text_content):
                number = match.group()
                if number not in metadata["numbers"]:
                    metadata["numbers"].append({
                        "value": number,
                        "length": len(number),
                        "context": text_content[:100]  # 保存上下文
                    })
                    logger.debug(f"[{request_id}] Found number: {number} (len={len(number)})")
            
            # 2. 提取所有日期（支持多种格式）
            date_patterns = [
                (r'(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日]?', 'YYYY-MM-DD'),
                (r'(\d{1,2})[月\-/](\d{1,2})[日\-/](\d{4})', 'MM-DD-YYYY'),
                (r'(\d{4})(\d{2})(\d{2})', 'YYYYMMDD'),
            ]
            
            for pattern, format_type in date_patterns:
                for match in re.finditer(pattern, text_content):
                    date_str = match.group()
                    if not any(d["value"] == date_str for d in metadata["dates"]):
                        metadata["dates"].append({
                            "value": date_str,
                            "format": format_type,
                            "context": text_content[:100]
                        })
                        logger.debug(f"[{request_id}] Found date: {date_str} (format={format_type})")
            
            # 3. 识别印章、签名等关键元素（使用更广泛的关键词）
            stamp_keywords = [
                '章', '印', '签', '戳', '专用', '财务', '税务', '公司',
                '电子', '数字', '认证', '授权', 'seal', 'stamp', 'signature',
                '发票专用章', '税务局'
            ]
            
            if any(keyword in text_content for keyword in stamp_keywords) or any(keyword in text_content.lower() for keyword in ["seal", "stamp", "signature"]):
                if text_content not in metadata["stamps"]:
                    metadata["stamps"].append(text_content)
                    logger.debug(f"[{request_id}] Found stamp-related: {text_content[:50]}")

            # 可能的开具机构
            issuer_match = re.findall(r'(?:税务局|税务管理局|国家税务总局|上海市税务局|北京市税务局)', text_content)
            for im in issuer_match:
                if im not in metadata["issuer_names"]:
                    metadata["issuer_names"].append(im)
            
            # 4. 尝试提取键值对（如：发票号码: xxx, 日期: xxx）
            kv_pattern = r'([^:：\n]{2,10})[：:]\s*([^\n]{1,50})'
            for match in re.finditer(kv_pattern, text_content):
                key = match.group(1).strip()
                value = match.group(2).strip()
                if key and value:
                    metadata["key_value_pairs"][key] = value
                    logger.debug(f"[{request_id}] Found key-value: {key} = {value}")

            # 电子发票标志
            if '电子发票' in text_content:
                metadata["flags"]["is_electronic_invoice"] = True
            
            # 5. 保存所有header文本
            if text_content not in metadata["headers"]:
                metadata["headers"].append(text_content)
    
    # 智能推断最可能的发票号和日期
    metadata["inferred"] = _infer_primary_fields(metadata, request_id)
    
    # 记录恢复结果
    total_items = (
        len(metadata["numbers"]) + 
        len(metadata["dates"]) + 
        len(metadata["stamps"]) +
        len(metadata["key_value_pairs"])
    )
    
    if total_items > 0:
        logger.info(
            f"[{request_id}] Extracted metadata: "
            f"numbers={len(metadata['numbers'])}, dates={len(metadata['dates'])}, "
            f"stamps={len(metadata['stamps'])}, kv_pairs={len(metadata['key_value_pairs'])}"
        )
    else:
        logger.warning(
            f"[{request_id}] No metadata extracted from discarded blocks"
        )
    
    return metadata


def _infer_primary_fields(metadata: dict, request_id: str) -> dict:
    """
    从提取的元数据中智能推断主要字段
    
    使用启发式规则：
    - 最长的数字串可能是发票号
    - 最近的日期可能是开票日期
    - 包含特定关键词的可能是特定字段
    """
    inferred = {
        "invoice_number": None,
        "invoice_code": None,
        "invoice_date": None,
        "primary_stamp": None,
    }
    
    # 从键值对中优先解析发票代码和发票号码
    for key, value in metadata.get("key_value_pairs", {}).items():
        if re.search(r"发票代码", key):
            m = re.search(r"\b\d{10,12}\b", value)
            if m:
                inferred["invoice_code"] = m.group(0)
        if re.search(r"发票(号码|编号)", key):
            m = re.search(r"\b\d{8,30}\b", value)
            if m:
                inferred["invoice_number"] = m.group(0)

    # 如果仍为空，则使用长度启发：优先选择19-20位（电子发票常见），其次12位（代码），再次8位（号码）
    if not inferred["invoice_number"] or not inferred["invoice_code"]:
        for num_info in metadata.get("numbers", []):
            n = num_info["value"]
            ln = num_info["length"]
            if not inferred["invoice_number"] and 18 <= ln <= 30:
                inferred["invoice_number"] = n
            if not inferred["invoice_code"] and 10 <= ln <= 12:
                inferred["invoice_code"] = n
            if inferred["invoice_number"] and inferred["invoice_code"]:
                break
    
    # 推断日期：优先选择YYYY-MM-DD格式的
    if metadata["dates"]:
        # 优先选择标准格式
        def _normalize_date(ds: str) -> str:
            try:
                # 2024年12月10日
                m = re.match(r"(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日]?", ds)
                if m:
                    y, mo, d = m.groups()
                    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
                m = re.match(r"(\d{4})(\d{2})(\d{2})", ds)
                if m:
                    y, mo, d = m.groups()
                    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            except Exception:
                pass
            return ds

        for date_info in metadata["dates"]:
            if date_info["format"] in ['YYYY-MM-DD', 'YYYYMMDD']:
                inferred["invoice_date"] = _normalize_date(date_info["value"]) or date_info["value"]
                logger.debug(f"[{request_id}] Inferred invoice date: {date_info['value']}")
                break
        
        # 如果没有标准格式，选择第一个
        if not inferred["invoice_date"] and metadata["dates"]:
            inferred["invoice_date"] = _normalize_date(metadata["dates"][0]["value"]) or metadata["dates"][0]["value"]
    
    # 推断主要印章：选择最短的（通常是主印章名称）
    if metadata["stamps"]:
        inferred["primary_stamp"] = min(metadata["stamps"], key=len)
        logger.debug(f"[{request_id}] Inferred primary stamp: {inferred['primary_stamp'][:50]}")
    
    # 从键值对中提取
    for key, value in metadata.get("key_value_pairs", {}).items():
        key_lower = key.lower()
        
        # 发票号相关
        if any(k in key_lower for k in ['发票', 'invoice', '号码', 'number', '编号']):
            if not inferred["invoice_number"] and re.match(r'\d{10,}', value):
                inferred["invoice_number"] = value
        
        # 日期相关
        if any(k in key_lower for k in ['日期', 'date', '时间', 'time']):
            if not inferred["invoice_date"]:
                inferred["invoice_date"] = value
    
    return inferred


def extract_invoice_metadata_from_content_list(content_list_str: str, request_id: str = "unknown") -> dict:
    """从 content_list 中补充提取发票元数据（发票代码、号码、日期、电子发票标志、印章关键词）。"""
    try:
        data = json.loads(content_list_str)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    m = {
        "numbers": [],
        "dates": [],
        "stamps": [],
        "headers": [],
        "key_value_pairs": {},
        "flags": {"is_electronic_invoice": False},
        "issuer_names": []
    }

    texts = []
    for item in data:
        if isinstance(item, dict):
            t = item.get("text")
            if t:
                texts.append(t)
            # 有些表格的 caption/footnote 也可能含关键信息
            if item.get("type") == "table":
                cap = item.get("table_caption") or ""
                foot = item.get("table_footnote") or ""
                if cap:
                    texts.append(cap)
                if foot:
                    texts.append(foot)

    blob = "\n".join(texts)
    if not blob:
        return {}

    # 电子发票
    if "电子发票" in blob:
        m["flags"]["is_electronic_invoice"] = True

    # 发票代码/号码键值模式
    for kv in re.findall(r"(发票代码|发票号码|发票编号|开票日期)[：:]\s*([\d年月日\-/]+)", blob):
        k, v = kv
        m["key_value_pairs"][k] = v
        if re.search(r"\d{8,30}", v):
            m["numbers"].append({"value": v, "length": len(v), "context": k})
        if "日期" in k:
            m["dates"].append({"value": v, "format": "YYYY-MM-DD", "context": k})

    # 游离数字与日期
    for num in re.findall(r"\b\d{8,30}\b", blob):
        m["numbers"].append({"value": num, "length": len(num), "context": "content_list"})
    for ds in re.findall(r"(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日]?", blob):
        y, mo, d = ds
        val = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        m["dates"].append({"value": val, "format": "YYYY-MM-DD", "context": "content_list"})

    # 印章/发行机构关键词
    for line in texts:
        if any(kw in line for kw in ["发票专用章", "税务局", "税务管理局"]):
            m["stamps"].append(line)
            for im in re.findall(r"(\S*税务局)", line):
                if im not in m["issuer_names"]:
                    m["issuer_names"].append(im)

    # 推断主要字段
    m["inferred"] = _infer_primary_fields(m, request_id)
    return m


def _merge_invoice_metadata(base: dict, sup: dict, request_id: str = "unknown") -> dict:
    """合并两份元数据，并重新推断主要字段。"""
    if not base:
        base = {}
    if not sup:
        return base

    merged = {
        "numbers": list(base.get("numbers", [])),
        "dates": list(base.get("dates", [])),
        "stamps": list(base.get("stamps", [])),
        "headers": list(base.get("headers", [])),
        "key_value_pairs": dict(base.get("key_value_pairs", {})),
        "flags": dict(base.get("flags", {})),
        "issuer_names": list(base.get("issuer_names", [])),
    }

    def _extend_unique(lst, items, key=lambda x: json.dumps(x, ensure_ascii=False)):
        seen = {key(i) for i in lst}
        for it in items:
            k = key(it)
            if k not in seen:
                lst.append(it)
                seen.add(k)

    _extend_unique(merged["numbers"], sup.get("numbers", []))
    _extend_unique(merged["dates"], sup.get("dates", []))
    _extend_unique(merged["stamps"], sup.get("stamps", []), key=lambda x: x)
    _extend_unique(merged["headers"], sup.get("headers", []), key=lambda x: x)

    merged["key_value_pairs"].update(sup.get("key_value_pairs", {}))
    merged["flags"].update(sup.get("flags", {}))
    _extend_unique(merged["issuer_names"], sup.get("issuer_names", []), key=lambda x: x)

    merged["inferred"] = _infer_primary_fields(merged, request_id)
    return merged


def enrich_content_list_with_metadata(content_list_str: str, metadata: dict, 
                                      request_id: str = "unknown") -> str:
    """
    将提取的元数据添加到content_list
    
    智能构建metadata block，包含所有识别到的信息
    
    Args:
        content_list_str: content_list JSON字符串
        metadata: 提取的元数据（新格式）
        request_id: 请求ID用于日志
        
    Returns:
        str: 增强后的content_list JSON字符串
    """
    try:
        content_list = json.loads(content_list_str)
    except json.JSONDecodeError:
        return content_list_str
    
    if not isinstance(content_list, list) or not metadata:
        return content_list_str
    
    # 构建文档元数据block
    doc_metadata_block = {
        "type": "document_metadata",
        "page_idx": 0,
    }
    
    # 使用推断的主要字段
    inferred = metadata.get("inferred", {})
    
    if inferred.get("invoice_number"):
        doc_metadata_block["invoice_number"] = inferred["invoice_number"]
    if inferred.get("invoice_code"):
        doc_metadata_block["invoice_code"] = inferred["invoice_code"]
    
    if inferred.get("invoice_date"):
        doc_metadata_block["invoice_date"] = inferred["invoice_date"]
    
    if inferred.get("primary_stamp"):
        doc_metadata_block["stamp_info"] = inferred["primary_stamp"]
    
    # 额外标志位与机构名称
    flags = metadata.get("flags", {})
    if flags:
        doc_metadata_block["flags"] = flags
    issuers = metadata.get("issuer_names", [])
    if issuers:
        doc_metadata_block["issuer_names"] = issuers
    
    # 添加所有提取的数字（可能包含其他重要编号）
    if metadata.get("numbers"):
        doc_metadata_block["extracted_numbers"] = [
            n["value"] for n in metadata["numbers"][:5]  # 最多保存5个
        ]
    
    # 添加所有日期
    if metadata.get("dates"):
        doc_metadata_block["extracted_dates"] = [
            d["value"] for d in metadata["dates"][:3]  # 最多保存3个
        ]
    
    # 添加键值对
    if metadata.get("key_value_pairs"):
        doc_metadata_block["key_value_pairs"] = metadata["key_value_pairs"]

    # 添加原始Header文本（通用做法）
    if metadata.get("headers"):
        doc_metadata_block["headers"] = metadata["headers"]
    
    # 只有在有实际内容时才插入
    if len(doc_metadata_block) > 2:  # 超过type和page_idx
        content_list.insert(0, doc_metadata_block)
        logger.info(
            f"[{request_id}] Enriched content_list with {len(doc_metadata_block)-2} metadata fields"
        )
        return json.dumps(content_list, ensure_ascii=False)
    
    return content_list_str


def enrich_markdown_with_metadata(markdown_content: str, metadata: dict, 
                                  request_id: str = "unknown") -> str:
    """
    仅在必要时补充缺失的元数据，保持MinerU原生输出的清晰格式
    
    策略：
    - 检查markdown中是否已包含关键信息
    - 只添加真正缺失的元数据
    - 使用简洁格式，不破坏原有结构
    
    Args:
        markdown_content: 原始markdown内容
        metadata: 提取的元数据
        request_id: 请求ID用于日志
        
    Returns:
        str: 增强后的markdown内容
    """
    if not markdown_content or not metadata:
        return markdown_content
    
    inferred = metadata.get("inferred", {})
    kv_pairs = metadata.get("key_value_pairs", {})
    
    # 检查markdown中是否已包含关键信息
    has_invoice_num = False
    has_date = False
    
    invoice_num = inferred.get("invoice_number")
    invoice_date = inferred.get("invoice_date")
    
    if invoice_num and invoice_num in markdown_content:
        has_invoice_num = True
    
    if invoice_date and invoice_date in markdown_content:
        has_date = True
    
    # 如果关键信息都已存在，不添加任何内容
    if has_invoice_num and has_date:
        logger.info(f"[{request_id}] Markdown already contains key metadata, no enrichment needed")
        return markdown_content
    
    # 只添加缺失的关键信息
    header_lines = []
    
    # 1. 首先添加所有原始Header文本（用户要求的通用做法）
    # 这些通常是VLM识别为header但被丢弃的内容，如"电子发票"、"税务局"等
    headers = metadata.get("headers", [])
    for header_text in headers:
        if header_text and header_text not in markdown_content:
            header_lines.append(header_text)
    
    # 2. 补充键值对（如果Header中未包含）
    # 优先从键值对中提取（格式更准确）
    for key, value in kv_pairs.items():
        key_lower = key.lower()
        
        # 添加缺失的发票号
        if not has_invoice_num and any(k in key_lower for k in ['号码', 'number', '编号']):
            if re.match(r'\d{10,}', value):
                # 检查是否已在headers中存在类似信息
                if not any(value in h for h in header_lines):
                    header_lines.append(f"{key}: {value}")
                    has_invoice_num = True
        
        # 添加缺失的日期
        if not has_date and any(k in key_lower for k in ['日期', 'date']):
            # 检查是否已在headers中存在类似信息
            if not any(value in h for h in header_lines):
                header_lines.append(f"{key}: {value}")
                has_date = True
    
    # 3. 补充推断信息（如果仍未包含）
    if not has_invoice_num and invoice_num:
        if not any(invoice_num in h for h in header_lines):
            header_lines.append(f"编号: {invoice_num}")
    
    if not has_date and invoice_date:
        if not any(invoice_date in h for h in header_lines):
            header_lines.append(f"日期: {invoice_date}")
    
    # 只在有缺失信息时才添加头部
    if header_lines:
        header_section = "\n".join(header_lines) + "\n\n"
        markdown_content = header_section + markdown_content
        logger.info(f"[{request_id}] Added {len(header_lines)} missing metadata fields to markdown")
    
    return markdown_content


def sanitize_filename(filename: str) -> str:
    """
    格式化化压缩文件的文件名
    移除路径遍历字符, 保留 Unicode 字母、数字、._- 
    禁止隐藏文件
    """
    sanitized = re.sub(r'[/\\\.]{2,}|[/\\]', '', filename)
    sanitized = re.sub(r'[^\w.-]', '_', sanitized, flags=re.UNICODE)
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    return sanitized or 'unnamed'


def cleanup_file(file_path: str) -> None:
    """清理临时 zip 文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"fail clean file {file_path}: {e}")


async def create_zip_response(
    output_dir: str,
    file_info: Dict[str, Any],
    actual_backend: str,
    actual_parse_method: str,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    request_id: str
) -> FileResponse:
    """创建ZIP格式的响应文件"""
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_results_")
    os.close(zip_fd)
    
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            pdf_name = file_info['stem']
            safe_pdf_name = sanitize_filename(pdf_name)
            
            # 确定解析目录
            if actual_backend.startswith("pipeline"):
                parse_dir = os.path.join(output_dir, pdf_name, actual_parse_method)
            else:
                parse_dir = os.path.join(output_dir, pdf_name, "vlm")

            # 检查解析目录是否存在
            if not os.path.exists(parse_dir):
                raise FileNotFoundError(f"Parse directory not found: {parse_dir}")

            # 写入文本类结果
            if return_md:
                path = os.path.join(parse_dir, f"{pdf_name}.md")
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}.md"))

            if return_middle_json:
                path = os.path.join(parse_dir, f"{pdf_name}_middle.json")
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_middle.json"))

            if return_model_output:
                # Both pipeline and VLM backends use _model.json
                path = os.path.join(parse_dir, f"{pdf_name}_model.json")
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.join(safe_pdf_name, os.path.basename(path)))

            if return_content_list:
                path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.join(safe_pdf_name, f"{safe_pdf_name}_content_list.json"))

            # 写入图片
            if return_images:
                images_dir = os.path.join(parse_dir, "images")
                if os.path.exists(images_dir):
                    # 扩展支持的图像格式
                    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff', 'tif', 'jp2']
                    image_paths = []
                    
                    for ext in image_extensions:
                        pattern = os.path.join(images_dir, f"*.{ext}")
                        image_paths.extend(glob(pattern))
                    
                    logger.info(f"[{request_id}] Found {len(image_paths)} images for ZIP in {images_dir}")
                    
                    added_count = 0
                    for image_path in image_paths:
                        try:
                            if os.path.exists(image_path) and os.access(image_path, os.R_OK):
                                zf.write(
                                    image_path, 
                                    arcname=os.path.join(safe_pdf_name, "images", os.path.basename(image_path))
                                )
                                added_count += 1
                            else:
                                logger.warning(f"[{request_id}] Cannot access image: {image_path}")
                        except Exception as e:
                            logger.warning(f"[{request_id}] Failed to add image to ZIP: {image_path}, error: {e}")
                    
                    logger.info(f"[{request_id}] Added {added_count} images to ZIP")

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="results.zip",
            background=BackgroundTask(cleanup_file, zip_path)
        )
    except Exception as e:
        # 如果创建ZIP文件失败，确保清理已创建的文件
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except:
            pass
        logger.error(f"[{request_id}] Failed to create ZIP response: {str(e)}")
        raise


@app.post(
    "/v2/parse/file",
    tags=["projects"],
    summary="Parse files using new API interface",
)
async def file_parse(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        output_dir: str = Form("./output"),
        backend: str = DEFAULT_BACKEND,
        parse_method: str = Form("auto"),
        lang: str = "ch",
        formula_enable: bool = True,
        table_enable: bool = True,
        return_md: bool = True,
        return_middle_json: bool = True,
        return_model_output: bool = False,
        return_content_list: bool = True,
        return_images: bool = True,
        response_format_zip: bool = False,
        start_page_id: int = 0,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 2,
        end_page_id: int = 99999,
        temperature: float = 0.0,
        #top_p: float = 0.0,
        #top_k: int = 1,
        #repetition_penalty: float = 1.05,
        server_url: Optional[str] = None,
):
    """
    Unified document parsing interface with enhanced robustness and resource management.
    Supports all document types and backends without concurrency restrictions.
    """
    temp_dir = None
    request_id = uuid.uuid4().hex[:REQUEST_ID_LENGTH]

    try:
        # Get file information
        file_info = get_file_info(file)
        logger.info(f"[{request_id}] Processing file: {file_info['filename']} with backend: {backend}")

        # Validate backend configuration
        try:
            backend_validation = validate_backend_config(backend, parse_method, server_url)
            actual_backend = backend_validation["backend"]
            actual_parse_method = backend_validation["parse_method"]
            backend_config = backend_validation["config"]
            
            # 为发票类文档优化VLM参数
            if actual_backend in VLM_BACKENDS_SET:
                vlm_recommendations = optimize_vlm_params_for_invoice(request_id)
                logger.debug(f"[{request_id}] VLM optimization: {vlm_recommendations}")

            if "warning" in backend_validation:
                logger.warning(f"[{request_id}] {backend_validation['warning']}")

        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": str(e)}
            )
        
        # 兼容无后缀或后缀不准确的上传文件：不在此处做严格后缀校验
        # 后续通过按字节识别并转换（read_fn/guess），以提升解析的健壮性
        file_info['suffix'] = file_info.get('suffix', '').lstrip('.').lower()

        # Read file content
        file_content = await file.read()

        # Validate file size
        if not validate_file_size(file_content):
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
                }
            )

        # Create unique temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"{TEMP_DIR_PREFIX}{actual_backend}_{request_id}_")
        logger.info(f"[{request_id}] Created temp directory: {temp_dir}")

        # Save uploaded file with explicit file handle management
        temp_file = os.path.join(temp_dir, file_info['filename'])
        try:
            with open(temp_file, "wb") as f:
                f.write(file_content)
                f.flush()  # Ensure data is written
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            logger.error(f"[{request_id}] Failed to save file: {str(e)}")
            # Clean up before raising exception
            await immediate_cleanup(temp_dir)
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        logger.info(f"[{request_id}] Saved {len(file_content)} bytes to {temp_file}")

        # Clear file content from memory immediately
        del file_content

        # 使用guess_suffix_by_path获取更准确的文件信息
        file_info = get_file_info_with_guess(temp_file)
        logger.info(f"[{request_id}] File info with guess: {file_info}")

        # Process file using aio_do_parse
        try:
            pdf_bytes = read_fn(temp_file)
        except Exception as e:
            logger.error(f"[{request_id}] Failed to read file: {str(e)}")
            # Clean up before raising exception
            await immediate_cleanup(temp_dir)
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        # Create output directory
        output_dir = os.path.join(temp_dir, OUTPUT_DIR_NAME)

        # Use server URL from validation result if available
        actual_server_url = backend_validation.get("server_url", server_url)

        # Prepare backend-specific parameters
        backend_params = prepare_backend_params(
            actual_backend,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            server_url=actual_server_url,
            # temperature/top_p/top_k/repetition_penalty 属于采样参数，推理阶段再传入，不应在引擎初始化传递
        )

        # Adjust language parameter for backend compatibility
        actual_lang = lang if backend_config.get("supports_ocr_lang", True) else "ch"

        # Set environment variables for VLM backend (critical for correct behavior)
        if actual_backend in VLM_BACKENDS_SET:
            os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable).lower()
            os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable).lower()
            logger.info(f"[{request_id}] Set VLM env vars: FORMULA={formula_enable}, TABLE={table_enable}")

        # Parse document
        logger.info(f"[{request_id}] Starting document parsing with backend: {actual_backend}, method: {actual_parse_method}")
        
        # Prepare backend name for aio_do_parse
        # VLM backends are passed as-is (e.g., "vlm-vllm-async-engine")
        # aio_do_parse will handle the backend name internally (removing "vlm-" prefix)
        backend_for_parse = actual_backend
        logger.debug(f"[{request_id}] Using backend for parse: {backend_for_parse}")
        
        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_info['stem']],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[actual_lang],
            backend=backend_for_parse,
            parse_method=actual_parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=True,  # 总是生成middle_json用于元数据提取
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            **backend_params
        )
        logger.info(f"[{request_id}] Document parsing completed")

        # Get result directory using backend-aware function
        result_dir = get_result_directory(output_dir, file_info['stem'], actual_backend, actual_parse_method)

        if not os.path.exists(result_dir):
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        # Build response data efficiently
        response_data = {
            "success": True,
            "backend": actual_backend,
            "parse_method": actual_parse_method,
            "version": __version__,
            "filename": file_info['filename'],
            "request_id": request_id,
        }

        # Add backend-specific information
        if "warning" in backend_validation:
            response_data["warning"] = backend_validation["warning"]
        
        # List files in result directory for debugging
        if os.path.exists(result_dir):
            files_in_dir = os.listdir(result_dir)
            logger.info(f"[{request_id}] Files in result_dir: {files_in_dir}")
        else:
            logger.warning(f"[{request_id}] Result directory does not exist: {result_dir}")
        
        # 提取被丢弃的发票元数据（需要在处理markdown和content_list之前）
        invoice_metadata = {}
        logger.info(f"[{request_id}] 检查诊断条件: return_content_list={return_content_list}, return_md={return_md}, return_middle_json={return_middle_json}")
        
        if return_content_list or return_md or return_middle_json:
            middle_json = get_infer_result("_middle.json", file_info['stem'], result_dir)
            logger.info(f"[{request_id}] Middle JSON loaded: {bool(middle_json)}, length={len(middle_json) if middle_json else 0}")
            
            if middle_json:
                # 如果需要返回middle_json，保存它
                if return_middle_json:
                    response_data["middle_json"] = middle_json
                
                # ========== 诊断日志开始 ==========
                logger.info(f"[{request_id}] 🔍 开始诊断VLM解析结果")
                
                try:
                    middle_data = json.loads(middle_json)
                    
                    for page_idx, page_info in enumerate(middle_data.get("pdf_info", [])):
                        discarded = page_info.get("discarded_blocks", [])
                        para_blocks = page_info.get("para_blocks", [])
                        
                        logger.info(f"[{request_id}] 📄 Page {page_idx}: "
                                   f"discarded_blocks={len(discarded)}, "
                                   f"para_blocks={len(para_blocks)}")
                        
                        # 记录discarded_blocks的详细信息
                        if discarded:
                            logger.info(f"[{request_id}] 🗑️  Discarded blocks详情:")
                            for idx, block in enumerate(discarded[:5]):  # 只记录前5个
                                block_type = block.get("type", "unknown")
                                # 尝试获取内容
                                content = ""
                                if "lines" in block and block["lines"]:
                                    first_line = block["lines"][0]
                                    if "spans" in first_line and first_line["spans"]:
                                        first_span = first_line["spans"][0]
                                        content = first_span.get("content", "")[:100]
                                
                                logger.info(f"[{request_id}]   [{idx}] type={block_type}, "
                                          f"content={content}")
                        else:
                            logger.warning(f"[{request_id}] ⚠️  Discarded blocks为空！")
                        
                        # 记录para_blocks的类型分布
                        if para_blocks:
                            type_counts = {}
                            for block in para_blocks:
                                block_type = block.get("type", "unknown")
                                type_counts[block_type] = type_counts.get(block_type, 0) + 1
                            
                            logger.info(f"[{request_id}] 📊 Para blocks类型分布: {type_counts}")
                            
                            # 记录前3个para_blocks的详细信息
                            logger.info(f"[{request_id}] 📝 前3个para blocks详情:")
                            for idx, block in enumerate(para_blocks[:3]):
                                block_type = block.get("type", "unknown")
                                # 尝试获取内容
                                content = ""
                                if "lines" in block and block["lines"]:
                                    first_line = block["lines"][0]
                                    if "spans" in first_line and first_line["spans"]:
                                        first_span = first_line["spans"][0]
                                        content = first_span.get("content", "")[:100]
                                elif "blocks" in block and block["blocks"]:
                                    # 处理嵌套结构（如list, table等）
                                    first_sub_block = block["blocks"][0]
                                    if "lines" in first_sub_block and first_sub_block["lines"]:
                                        first_line = first_sub_block["lines"][0]
                                        if "spans" in first_line and first_line["spans"]:
                                            first_span = first_line["spans"][0]
                                            content = first_span.get("content", "")[:100]
                                
                                logger.info(f"[{request_id}]   [{idx}] type={block_type}, "
                                          f"content={content}")
                        
                        # 在discarded_blocks和para_blocks中搜索发票号码和日期
                        logger.info(f"[{request_id}] 🔎 搜索发票号码和日期...")
                        
                        all_blocks = discarded + para_blocks
                        found_numbers = []
                        found_dates = []
                        
                        for block in all_blocks:
                            block_type = block.get("type", "unknown")
                            # 提取所有文本内容
                            texts = []
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            if "content" in span:
                                                texts.append(span["content"])
                            elif "blocks" in block:
                                for sub_block in block["blocks"]:
                                    if "lines" in sub_block:
                                        for line in sub_block["lines"]:
                                            if "spans" in line:
                                                for span in line["spans"]:
                                                    if "content" in span:
                                                        texts.append(span["content"])
                            
                            full_text = " ".join(texts)
                            
                            # 搜索长数字串（可能是发票号码）
                            import re
                            numbers = re.findall(r'\b\d{10,}\b', full_text)
                            for num in numbers:
                                found_numbers.append({
                                    "value": num,
                                    "block_type": block_type,
                                    "context": full_text[:100]
                                })
                            
                            # 搜索日期
                            dates = re.findall(r'(\d{4})[年\-/](\d{1,2})[月\-/](\d{1,2})[日]?', full_text)
                            for date_match in dates:
                                y, m, d = date_match
                                date_str = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
                                found_dates.append({
                                    "value": date_str,
                                    "block_type": block_type,
                                    "context": full_text[:100]
                                })
                        
                        if found_numbers:
                            logger.info(f"[{request_id}] 🔢 找到 {len(found_numbers)} 个长数字串:")
                            for num_info in found_numbers[:3]:  # 只显示前3个
                                logger.info(f"[{request_id}]   - {num_info['value']} "
                                          f"(type={num_info['block_type']})")
                        else:
                            logger.warning(f"[{request_id}] ⚠️  未找到长数字串（发票号码）")
                        
                        if found_dates:
                            logger.info(f"[{request_id}] 📅 找到 {len(found_dates)} 个日期:")
                            for date_info in found_dates[:3]:  # 只显示前3个
                                logger.info(f"[{request_id}]   - {date_info['value']} "
                                          f"(type={date_info['block_type']})")
                        else:
                            logger.warning(f"[{request_id}] ⚠️  未找到日期格式")
                        
                        # 只诊断第一页
                        if page_idx == 0:
                            break
                
                except Exception as e:
                    logger.error(f"[{request_id}] ❌ 诊断过程出错: {e}")
                
                logger.info(f"[{request_id}] 🔍 诊断完成")
                # ========== 诊断日志结束 ==========
                
                # 提取发票元数据 - 使用增强版本，检查所有blocks
                logger.info(f"[{request_id}] Using enhanced metadata extraction (checks all blocks)")
                invoice_metadata = extract_invoice_metadata_from_all_blocks(middle_json, request_id)
                
                # 检查是否提取到有效元数据
                if invoice_metadata:
                    inferred = invoice_metadata.get("inferred", {})
                    has_metadata = (
                        bool(inferred.get("invoice_number")) or
                        bool(inferred.get("invoice_date")) or
                        bool(inferred.get("primary_stamp")) or
                        len(invoice_metadata.get("numbers", [])) > 0 or
                        len(invoice_metadata.get("dates", [])) > 0 or
                        len(invoice_metadata.get("key_value_pairs", {})) > 0
                    )
                    
                    if has_metadata:
                        logger.info(
                            f"[{request_id}] Extracted metadata: "
                            f"inferred_number={bool(inferred.get('invoice_number'))}, "
                            f"inferred_date={bool(inferred.get('invoice_date'))}, "
                            f"total_numbers={len(invoice_metadata.get('numbers', []))}, "
                            f"total_dates={len(invoice_metadata.get('dates', []))}, "
                            f"kv_pairs={len(invoice_metadata.get('key_value_pairs', {}))}"
                        )



        # Get content_list并增强（先处理，以便合并元数据）
        merged_meta = invoice_metadata  # 默认使用从middle_json提取的元数据
        
        if return_model_output:
            # Both pipeline and VLM backends use _model.json
            model_output = get_infer_result("_model.json", file_info['stem'], result_dir)
            if model_output:
                response_data["model_output"] = model_output

        if return_content_list:
            content_list = get_infer_result("_content_list.json", file_info['stem'], result_dir)
            if content_list:
                # 先进行发票专用增强
                content_list = enhance_content_list_for_invoice(content_list, request_id)
                
                # 进一步从 content_list 中补充元数据并与 middle_json 的元数据合并
                try:
                    sup_meta = extract_invoice_metadata_from_content_list(content_list, request_id)
                except Exception:
                    sup_meta = {}
                merged_meta = _merge_invoice_metadata(invoice_metadata, sup_meta, request_id) if (invoice_metadata or sup_meta) else invoice_metadata

                # 插入文档元数据块，并将主要字段暴露到响应中
                if merged_meta:
                    content_list = enrich_content_list_with_metadata(
                        content_list, merged_meta, request_id
                    )
                    response_data["document_metadata"] = merged_meta.get("inferred", {})
                    response_data["document_metadata_flags"] = merged_meta.get("flags", {})
                
                # 验证和诊断表格内容
                content_list, table_diagnostics = validate_and_enhance_table_content(
                    content_list, 
                    request_id
                )
                response_data["content_list"] = content_list
                
                # 添加表格诊断信息
                if table_diagnostics:
                    if "diagnostics" not in response_data:
                        response_data["diagnostics"] = {}
                    response_data["diagnostics"]["tables"] = table_diagnostics
        
        # Get markdown content并增强（使用合并后的元数据）
        if return_md:
            md_content = get_infer_result(".md", file_info['stem'], result_dir)
            if md_content:
                # 使用合并后的完整元数据增强markdown
                if merged_meta:
                    logger.info(f"[{request_id}] Enriching markdown with merged metadata")
                    md_content = enrich_markdown_with_metadata(
                        md_content, merged_meta, request_id
                    )
                response_data["markdown"] = md_content
                
                # Extract page count safely
                try:
                    content_data = json.loads(content_list)
                    if content_data and isinstance(content_data, list):
                        response_data["pages"] = content_data[-1].get('page_idx', 0) + 1
                    else:
                        response_data["pages"] = 0
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"[{request_id}] Failed to extract page count: {str(e)}")
                    response_data["pages"] = 0

        # Get images efficiently
        if return_images:
            image_dir = os.path.join(result_dir, IMAGE_DIR_NAME)
            if os.path.exists(image_dir):
                try:
                    # 使用增强的图像处理逻辑
                    images_data = await process_images_async(
                        image_dir, 
                        backend=actual_backend,
                        request_id=request_id
                    )
                    
                    if images_data:
                        response_data["images"] = images_data
                        logger.info(f"[{request_id}] Processed {len(images_data)} images for response")

                        # 使用增强的图片替换函数
                        if "markdown" in response_data and response_data["markdown"]:
                            original_md = response_data["markdown"]
                            response_data["markdown"] = replace_markdown_images(
                                original_md, 
                                images_data, 
                                request_id
                            )
                        
                        # 验证并诊断图片引用
                        response_data = validate_and_fix_images(response_data, request_id)
                    else:
                        logger.warning(f"[{request_id}] No images processed from {image_dir}")

                except Exception as e:
                    logger.error(f"[{request_id}] Error processing images: {str(e)}", exc_info=True)
                    # Continue without images rather than failing

        # Schedule delayed cleanup
        background_tasks.add_task(delayed_cleanup, temp_dir)
        logger.info(f"[{request_id}] Processing completed successfully")

        # 根据 response_format_zip 决定返回类型
        if response_format_zip:
            return await create_zip_response(
                output_dir=output_dir,
                file_info=file_info,
                actual_backend=actual_backend,
                actual_parse_method=actual_parse_method,
                return_md=return_md,
                return_middle_json=return_middle_json,
                return_model_output=return_model_output,
                return_content_list=return_content_list,
                return_images=return_images,
                request_id=request_id
            )
        else:
            return JSONResponse(content=response_data, status_code=200)

    except HTTPException:
        # Re-raise HTTP exceptions
        if temp_dir:
            await immediate_cleanup(temp_dir)
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Error processing file {file.filename}: {str(e)}")
        if temp_dir:
            await immediate_cleanup(temp_dir)
        return JSONResponse(
            content={
                "success": False,
                "error": f"Failed to process file: {str(e)}",
                "request_id": request_id
            },
            status_code=500
        )



@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    try:
        import psutil
        async with connection_lock:
            active_conn_count = len(active_connections)

        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "active_connections": active_conn_count,
            "version": __version__,
            "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
            "supported_extensions": SUPPORTED_EXTENSIONS
        }
    except ImportError:
        async with connection_lock:
            active_conn_count = len(active_connections)

        return {
            "active_connections": active_conn_count,
            "version": __version__,
            "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "note": "Install psutil for system metrics"
        }

@app.post("/admin/cleanup")
async def force_cleanup():
    """Force cleanup of resources (admin endpoint)"""
    try:
        # Force garbage collection
        import gc
        gc.collect()

        # Get connection count
        async with connection_lock:
            conn_count = len(active_connections)

        return {
            "success": True,
            "message": "Cleanup completed",
            "active_connections": conn_count
        }
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/mode/switch")
async def get_mode_switch_info():
    """Get information about switching modes"""
    return {
        "current_mode": RUNTIME_MODE,
        "active_backends": ACTIVE_BACKENDS,
        "switch_instructions": {
            "to_pipeline": {
                "command": "MINERU_MODE=pipeline python mineru.py",
                "description": "Switch to traditional OCR-based processing",
                "features": ["formula", "table", "multiple_parse_methods", "ocr_languages"]
            },
            "to_vlm": {
                "command": "MINERU_MODE=vlm MINERU_VLM_BACKEND=vlm-vllm-async-engine python new.py",
                "description": "Switch to VLM-based processing",
                "features": ["advanced_reasoning", "complex_layouts", "generation_parameters"],
                "backends": VLM_BACKENDS,
                "note": "May require GPU and/or server URL"
            }
        },
        "environment_variables": {
            "MINERU_MODE": {
                "description": "Set processing mode",
                "values": ["pipeline", "vlm"],
                "current": RUNTIME_MODE
            },
            "MINERU_VLM_BACKEND": {
                "description": "Choose VLM backend when in VLM mode",
                "values": VLM_BACKENDS,
                "current": VLM_BACKEND if RUNTIME_MODE == "vlm" else None,
                "note": "Use 'vlm-' prefix format (e.g., vlm-vllm-async-engine)"
            },
            "MINERU_SERVER_URL": {
                "description": "Server URL for VLM client mode",
                "example": "http://127.0.0.1:30000",
                "current": SERVER_URL
            }
        }
    }

if __name__ == "__main__":
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Display startup configuration
    logger.info("=" * 60)
    logger.info("MinerU FastAPI Server Starting")
    logger.info("=" * 60)
    logger.info(f"Runtime Mode: {RUNTIME_MODE.upper()}")
    logger.info(f"Active Backends: {ACTIVE_BACKENDS}")
    logger.info(f"Default Backend: {DEFAULT_BACKEND}")

    if RUNTIME_MODE == "vlm":
        logger.info(f"VLM Backend: {VLM_BACKEND}")
        if SERVER_URL:
            logger.info(f"Server URL: {SERVER_URL}")
        else:
            logger.warning("No server URL configured (MINERU_SERVER_URL)")

    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.1f}MB")
    logger.info(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
    logger.info("=" * 60)
    logger.info("Server endpoints:")
    logger.info("  Main API: http://0.0.0.0:7434")
    logger.info("  Parse endpoint: http://0.0.0.0:7434/v2/parse/file")
    logger.info("  Documentation: http://0.0.0.0:7434/docs")
    logger.info("  Health check: http://0.0.0.0:7434/health")
    logger.info("  Backend info: http://0.0.0.0:7434/backends")
    logger.info("  Mode switch info: http://0.0.0.0:7434/mode/switch")
    logger.info("  Metrics: http://0.0.0.0:7434/metrics")
    logger.info("=" * 60)

    # Mode switching instructions
    logger.info("To switch modes:")
    logger.info("  Pipeline mode: MINERU_MODE=pipeline python new.py")
    logger.info("  VLM mode: MINERU_MODE=vlm MINERU_VLM_BACKEND=vlm-vllm-async-engine python new.py")
    logger.info("=" * 60)

    # Validate configuration
    config_valid = True
    if RUNTIME_MODE == "vlm":
        if DEFAULT_BACKEND not in VLM_BACKENDS:
            logger.error(f"ERROR: Invalid VLM backend '{DEFAULT_BACKEND}'. Valid options: {VLM_BACKENDS}")
            config_valid = False

    if not config_valid:
        logger.error("Configuration validation failed. Please fix the above errors and restart.")
        exit(1)

    logger.info("Configuration validated successfully. Starting server...")

    # System-level TCP optimization recommendations
    logger.info("For optimal connection handling, consider these system settings:")
    logger.info("  echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse")
    logger.info("  echo 1 > /proc/sys/net/ipv4/tcp_fin_timeout")
    logger.info("  echo 65536 > /proc/sys/net/core/somaxconn")

    # Start server without concurrency restrictions
    server_config = {
        "host": "0.0.0.0",
        "port": 7434,
        "access_log": False,
        "use_colors": True,
        "workers": 1,
    }

    # Add optional parameters if supported (without concurrency limits)
    try:
        import inspect
        uvicorn_run_params = inspect.signature(uvicorn.run).parameters

        if "timeout_keep_alive" in uvicorn_run_params:
            server_config["timeout_keep_alive"] = 30

        if "loop" in uvicorn_run_params:
            server_config["loop"] = "asyncio"

        if "http" in uvicorn_run_params:
            server_config["http"] = "h11"

        logger.info(f"Starting server with config: {list(server_config.keys())}")
        uvicorn.run(app, **server_config)

    except Exception as e:
        # Ultimate fallback - minimal configuration
        logger.warning(f"Using minimal server configuration due to: {e}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7434,
            access_log=False,
            workers=1,
        )


