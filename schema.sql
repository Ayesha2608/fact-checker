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
);

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
);

CREATE TABLE IF NOT EXISTS Feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_check_id INTEGER,
    helpful INTEGER NOT NULL,
    comment TEXT,
    date TEXT NOT NULL,
    FOREIGN KEY (fact_check_id) REFERENCES FactChecks(id)
);

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
);

CREATE TABLE IF NOT EXISTS Cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    date TEXT NOT NULL
);
