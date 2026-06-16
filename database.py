"""SQLite persistence for fact checks, citations, feedback, history, and cache."""

import json
import logging
import sqlite3
from collections import Counter

from utils import current_timestamp


LOGGER = logging.getLogger(__name__)


class DatabaseManager:
    """Encapsulate all SQLite access for the chatbot."""

    def __init__(self, path="fact_checker.db"):
        self.path = path
        self._memory_connection = None
        self.initialize()

    def connect(self):
        """Open a SQLite connection."""
        if self.path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(self.path)
            return self._memory_connection
        return sqlite3.connect(self.path)

    def initialize(self):
        """Create required tables."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS FactChecks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    date TEXT NOT NULL,
                    status TEXT DEFAULT 'VERIFIED',
                    verdict TEXT NOT NULL,
                    confidence INTEGER,
                    confidence_level TEXT NOT NULL,
                    category TEXT DEFAULT 'General',
                    quality_score INTEGER DEFAULT 0,
                    keywords_json TEXT DEFAULT '[]',
                    explanation TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_check_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    domain TEXT,
                    reliability INTEGER,
                    stance TEXT,
                    evidence TEXT,
                    FOREIGN KEY (fact_check_id) REFERENCES FactChecks(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_check_id INTEGER,
                    helpful INTEGER NOT NULL,
                    comment TEXT,
                    date TEXT NOT NULL,
                    FOREIGN KEY (fact_check_id) REFERENCES FactChecks(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS History (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_check_id INTEGER,
                    query TEXT NOT NULL,
                    date TEXT NOT NULL,
                    category TEXT,
                    status TEXT,
                    verdict TEXT,
                    confidence INTEGER,
                    quality_score INTEGER,
                    sources_json TEXT,
                    FOREIGN KEY (fact_check_id) REFERENCES FactChecks(id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    date TEXT NOT NULL
                )
                """
            )
            self._relax_factchecks_confidence(cursor)
            self._ensure_column(cursor, "FactChecks", "status", "TEXT DEFAULT 'VERIFIED'")
            self._ensure_column(cursor, "FactChecks", "category", "TEXT DEFAULT 'General'")
            self._ensure_column(cursor, "FactChecks", "quality_score", "INTEGER DEFAULT 0")
            self._ensure_column(cursor, "FactChecks", "keywords_json", "TEXT DEFAULT '[]'")
            self._ensure_column(cursor, "History", "category", "TEXT")
            self._ensure_column(cursor, "History", "status", "TEXT")
            self._ensure_column(cursor, "History", "quality_score", "INTEGER")
            connection.commit()

    def _ensure_column(self, cursor, table, column, definition):
        """Add a column to an existing SQLite table if it is missing."""
        cursor.execute("PRAGMA table_info({0})".format(table))
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute("ALTER TABLE {0} ADD COLUMN {1} {2}".format(table, column, definition))

    def _relax_factchecks_confidence(self, cursor):
        """Migrate older databases where confidence was incorrectly NOT NULL."""
        cursor.execute("PRAGMA table_info(FactChecks)")
        columns = cursor.fetchall()
        confidence = [row for row in columns if row[1] == "confidence"]
        if not confidence or not confidence[0][3]:
            return
        cursor.execute("ALTER TABLE FactChecks RENAME TO FactChecks_old")
        cursor.execute(
            """
            CREATE TABLE FactChecks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                date TEXT NOT NULL,
                status TEXT DEFAULT 'VERIFIED',
                verdict TEXT NOT NULL,
                confidence INTEGER,
                confidence_level TEXT NOT NULL,
                category TEXT DEFAULT 'General',
                quality_score INTEGER DEFAULT 0,
                keywords_json TEXT DEFAULT '[]',
                explanation TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO FactChecks
            (id, query, date, status, verdict, confidence, confidence_level, category, quality_score, keywords_json, explanation)
            SELECT
                id,
                query,
                date,
                COALESCE(status, 'VERIFIED'),
                verdict,
                confidence,
                confidence_level,
                COALESCE(category, 'General'),
                COALESCE(quality_score, 0),
                COALESCE(keywords_json, '[]'),
                explanation
            FROM FactChecks_old
            """
        )
        cursor.execute("DROP TABLE FactChecks_old")

    def save_fact_check(self, result):
        """Save a full verification result and return its ID."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO FactChecks
                (query, date, status, verdict, confidence, confidence_level, category, quality_score, keywords_json, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("query", ""),
                    current_timestamp(),
                    result.get("status", "VERIFIED"),
                    result.get("verdict", "Insufficient Evidence"),
                    self._confidence_value(result),
                    result.get("confidence", {}).get("level", "Very Low"),
                    result.get("processed_query", {}).get("category", "General"),
                    int(result.get("quality_metrics", {}).get("overall_verification_quality", 0)),
                    json.dumps(result.get("processed_query", {}).get("keywords", [])),
                    result.get("explanation", ""),
                ),
            )
            fact_check_id = cursor.lastrowid
            for source in result.get("sources", []):
                cursor.execute(
                    """
                    INSERT INTO Sources
                    (fact_check_id, url, title, domain, reliability, stance, evidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fact_check_id,
                        source.get("url", ""),
                        source.get("title", ""),
                        source.get("domain", ""),
                        int(source.get("reliability", 0)),
                        source.get("stance", ""),
                        source.get("evidence", ""),
                    ),
                )
            cursor.execute(
                """
                INSERT INTO History
                (fact_check_id, query, date, category, status, verdict, confidence, quality_score, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_check_id,
                    result.get("query", ""),
                    current_timestamp(),
                    result.get("processed_query", {}).get("category", "General"),
                    result.get("status", "VERIFIED"),
                    result.get("verdict", "Insufficient Evidence"),
                    self._confidence_value(result),
                    int(result.get("quality_metrics", {}).get("overall_verification_quality", 0)),
                    json.dumps(result.get("citations", [])),
                ),
            )
            connection.commit()
            return fact_check_id

    def list_history(self, limit=20):
        """Return recent history rows."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT id, query, date, category, status, verdict, confidence, quality_score
                FROM History
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            return cursor.fetchall()

    def search_history(self, keyword):
        """Search saved history records."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT id, query, date, category, status, verdict, confidence, quality_score
                FROM History
                WHERE query LIKE ?
                ORDER BY id DESC
                """,
                ("%" + keyword + "%",),
            )
            return cursor.fetchall()

    def analytics_summary(self):
        """Return dashboard statistics for project demonstration."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*), AVG(confidence), AVG(quality_score) FROM History")
            total, avg_confidence, avg_quality = cursor.fetchone()
            cursor.execute("SELECT category, COUNT(*) FROM History GROUP BY category ORDER BY COUNT(*) DESC LIMIT 5")
            categories = cursor.fetchall()
            cursor.execute("SELECT verdict, COUNT(*) FROM History GROUP BY verdict ORDER BY COUNT(*) DESC")
            verdicts = cursor.fetchall()
            cursor.execute("SELECT domain, COUNT(*) FROM Sources GROUP BY domain ORDER BY COUNT(*) DESC LIMIT 5")
            sources = cursor.fetchall()
            cursor.execute("SELECT date FROM History ORDER BY id DESC LIMIT 200")
            dates = cursor.fetchall()
            cursor.execute("SELECT keywords_json FROM FactChecks ORDER BY id DESC LIMIT 200")
            keyword_rows = cursor.fetchall()
        daily = Counter(row[0][:10] for row in dates if row and row[0])
        keywords = Counter()
        for row in keyword_rows:
            try:
                keywords.update(json.loads(row[0] or "[]"))
            except Exception:
                continue
        return {
            "total_fact_checks": total or 0,
            "average_confidence": int(round(avg_confidence or 0)),
            "average_quality": int(round(avg_quality or 0)),
            "most_common_categories": categories,
            "verification_statistics": verdicts,
            "top_sources": sources,
            "daily_usage": daily.most_common(7),
            "most_frequent_keywords": keywords.most_common(10),
        }

    def add_feedback(self, fact_check_id, helpful, comment=""):
        """Store user feedback for a result."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO Feedback (fact_check_id, helpful, comment, date)
                VALUES (?, ?, ?, ?)
                """,
                (fact_check_id, 1 if helpful else 0, comment, current_timestamp()),
            )
            connection.commit()

    def feedback_statistics(self):
        """Return helpful/not-helpful counts."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT helpful, COUNT(*) FROM Feedback GROUP BY helpful")
            rows = dict(cursor.fetchall())
        return {"helpful": rows.get(1, 0), "not_helpful": rows.get(0, 0)}

    def cache_get(self, key):
        """Return cached JSON data if available."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT value FROM Cache WHERE key = ?", (key,))
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def cache_set(self, key, value):
        """Save JSON-serializable cache data."""
        with self.connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO Cache (key, value, date)
                VALUES (?, ?, ?)
                """,
                (key, json.dumps(value), current_timestamp()),
            )
            connection.commit()

    def _confidence_value(self, result):
        value = result.get("confidence", {}).get("percentage")
        if value is None:
            return None
        return int(value)
