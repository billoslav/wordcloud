"""
Simple Flask API wrapper for generating wordcloud images.
Optional dependency: Flask. Only enabled when installed.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from threading import Thread
from typing import Dict, Any

try:
    from flask import Flask, jsonify, request, send_file # type: ignore
    FLASK_AVAILABLE = True
except ImportError:  # pragma: no cover
    FLASK_AVAILABLE = False

from wordcloud import Wordcloud
from wordcloud.utils import STRATEGIES
from wordcloud.utils.visualization import COLOR_THEMES
from wordcloud.utils.helpers import create_folder


def validate_api_params(data: Dict[str, Any]) -> tuple[Dict[str, Any], str | None]:
    """
    Validate and normalize API parameters.
    
    Returns:
        Tuple of (validated_params, error_message)
        If error_message is not None, params should be ignored.
    """
    params: Dict[str, Any] = {}
    
    # Required: text
    text = data.get("text")
    if not text:
        return {}, "text is required"
    if not isinstance(text, str) or not text.strip():
        return {}, "text must be a non-empty string"
    params["text"] = text
    
    # Optional: dimensions (with validation)
    width = data.get("width", 600)
    height = data.get("height", 338)
    try:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return {}, "width and height must be positive integers"
        if width > 10000 or height > 10000:
            return {}, "width and height must be <= 10000"
        params["width"] = width
        params["height"] = height
    except (ValueError, TypeError):
        return {}, "width and height must be integers"
    
    # Optional: word limits
    max_words = data.get("max_words", 200)
    try:
        max_words = int(max_words)
        if max_words <= 0:
            return {}, "max_words must be positive"
        if max_words > 10000:
            return {}, "max_words must be <= 10000"
        params["max_words"] = max_words
    except (ValueError, TypeError):
        return {}, "max_words must be an integer"
    
    # Optional: min_word_length
    if "min_word_length" in data:
        try:
            min_word_length = int(data["min_word_length"])
            if min_word_length <= 0:
                return {}, "min_word_length must be positive"
            params["min_word_length"] = min_word_length
        except (ValueError, TypeError):
            return {}, "min_word_length must be an integer"
    
    # Optional: stopwords
    if "stopwords" in data:
        stopwords = data["stopwords"]
        if isinstance(stopwords, str):
            params["stopwords"] = [s.strip() for s in stopwords.split(",") if s.strip()]
        elif isinstance(stopwords, list):
            params["stopwords"] = [str(s).strip() for s in stopwords if str(s).strip()]
        else:
            return {}, "stopwords must be a string or list"
    
    # Optional: placement strategy
    if "strategy" in data:
        strategy = data["strategy"]
        if strategy not in STRATEGIES:
            return {}, f"strategy must be one of: {', '.join(STRATEGIES)}"
        params["place_strategy"] = strategy
    
    # Optional: background color
    if "background_color" in data:
        params["background_color"] = str(data["background_color"])
    
    # Optional: color theme
    if "color_theme" in data:
        theme = data["color_theme"]
        if theme not in COLOR_THEMES:
            return {}, f"color_theme must be one of: {', '.join(COLOR_THEMES.keys())}"
        params["color_theme"] = theme
    
    # Optional: font path
    if "font_path" in data:
        font_path = Path(data["font_path"])
        if not font_path.exists():
            return {}, f"font_path not found: {font_path}"
        params["font_path"] = str(font_path)
    
    # Optional: black_white
    if "black_white" in data:
        params["black_white"] = bool(data["black_white"])
    
    return params, None


def create_app() -> "Flask":
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask is not installed. Install with `pip install flask` to use the API.")

    app = Flask(__name__)
    jobs: Dict[str, Dict[str, Any]] = {}
    tmp_root = create_folder(Path(tempfile.gettempdir()) / "wordcloud_api")

    def process_job(job_id: str, params: Dict[str, Any]) -> None:
        try:
            text = params.pop("text")
            color_theme = params.pop("color_theme", None)
            
            # Disable config loading in API to avoid issues with test environments
            params.setdefault('use_config', False)
            wc = Wordcloud(**params)
            wc.generate(text, color_theme=(color_theme if not params.get("black_white", False) else None))
            
            img = wc.draw_image(save_file=False)
            out_dir = create_folder(tmp_root / job_id)
            out_path = out_dir / "wordcloud.png"
            img.save(out_path)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = str(out_path)
        except Exception as exc:  # pragma: no cover - defensive
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)

    @app.route("/api/version", methods=["GET"])
    def version():
        """Get API version information."""
        return jsonify({
            "version": "0.1",
            "name": "wordcloud-api",
            "strategies": STRATEGIES,
            "color_themes": list(COLOR_THEMES.keys())
        })

    @app.route("/api/generate", methods=["POST"])
    def generate():
        """Generate a wordcloud asynchronously."""
        if not request.is_json:
            return jsonify({"error": "JSON body required"}), 400
        
        data = request.get_json() or {}
        params, error = validate_api_params(data)
        
        if error:
            return jsonify({"error": error}), 400
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "pending"}
        Thread(target=process_job, args=(job_id, params), daemon=True).start()
        return jsonify({"job_id": job_id, "status": "pending"}), 202

    @app.route("/api/status/<job_id>", methods=["GET"])
    def status(job_id: str):
        """Get job status."""
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job)

    @app.route("/api/download/<job_id>", methods=["GET"])
    def download(job_id: str):
        """Download completed wordcloud image."""
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        
        if job.get("status") == "failed":
            return jsonify({"error": job.get("error", "Job failed")}), 400
        
        if job.get("status") != "completed":
            return jsonify({"error": "Job not ready", "status": job.get("status")}), 202
        
        result_path = job.get("result")
        if not result_path or not Path(result_path).exists():
            return jsonify({"error": "Result file not found"}), 404
        
        return send_file(result_path, mimetype="image/png", as_attachment=True)

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    return app


def run_api(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """Run the Flask API server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)

