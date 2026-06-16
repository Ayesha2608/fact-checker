"""Standard-library web server for the fact-checking frontend."""

import os
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from database import DatabaseManager
from nlp_evaluation import NLPEvaluator
from verifier import FactCheckingEngine


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


class FactCheckWebApp(BaseHTTPRequestHandler):
    """Serve the frontend and expose JSON endpoints without external packages."""

    engine = FactCheckingEngine()
    database = DatabaseManager()
    evaluator = NLPEvaluator()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(FRONTEND_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            target = FRONTEND_DIR / parsed.path.removeprefix("/static/")
            self._serve_file(target)
            return
        if parsed.path == "/api/health":
            self._json({"status": "ok", "message": "Fact-checking backend is ready"})
            return
        if parsed.path == "/api/history":
            params = parse_qs(parsed.query)
            limit = self._int(params.get("limit", ["10"])[0], 10)
            rows = self.database.list_history(limit=limit)
            self._json({"history": [self._history_row(row) for row in rows]})
            return
        self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/verify":
                claim = (payload.get("claim") or "").strip()
                mode = payload.get("mode") or "general"
                if not claim:
                    self._json({"error": "Claim text is required"}, status=400)
                    return
                result = self.engine.verify_claim(claim, mode=mode)
                self._json({"result": result})
                return
            if parsed.path == "/api/evaluate":
                self._json({"metrics": self.evaluator.evaluate()})
                return
            if parsed.path == "/api/pipeline":
                text = (payload.get("text") or "").strip()
                if not text:
                    self._json({"error": "Text is required"}, status=400)
                    return
                self._json({"pipeline": self.evaluator.demo_pipeline(text)})
                return
            self._not_found()
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def log_message(self, format_string, *args):
        """Keep terminal output focused during demos."""
        return

    def _serve_file(self, path):
        try:
            resolved = path.resolve()
            if FRONTEND_DIR not in resolved.parents and resolved != FRONTEND_DIR:
                self._not_found()
                return
            if not resolved.exists() or not resolved.is_file():
                self._not_found()
                return
            content = resolved.read_bytes()
            content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _read_json(self):
        length = self._int(self.headers.get("Content-Length", "0"), 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw or "{}")

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self._json({"error": "Not found"}, status=404)

    def _history_row(self, row):
        return {
            "id": row[0],
            "query": row[1],
            "date": row[2],
            "category": row[3],
            "status": row[4],
            "verdict": row[5],
            "confidence": row[6],
            "quality_score": row[7],
        }

    def _int(self, value, default):
        try:
            return int(value)
        except Exception:
            return default


def run(host="0.0.0.0", port=None):
    if port is None:
        port = int(os.environ.get("PORT", "8001"))
    server = ThreadingHTTPServer((host, port), FactCheckWebApp)
    print("Fact-checking web app running at http://{0}:{1}".format(host, port))
    server.serve_forever()


if __name__ == "__main__":
    run()
