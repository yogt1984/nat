"""
arXiv Paper Reader — fetches recent papers and extracts signal ideas.

Uses stdlib urllib (no dependency) to query arXiv Atom API.
Claude extracts testable signal ideas from abstracts.
Results stored in SQLite arxiv_papers table for LLM generator context.

Usage:
  python scripts/agent/arxiv_reader.py scan         # Fetch + process papers
  python scripts/agent/arxiv_reader.py list          # Show processed papers
  python scripts/agent/arxiv_reader.py ideas         # Show extracted ideas
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger("nat.arxiv")

ROOT = Path(__file__).resolve().parent.parent.parent

SEARCH_QUERIES = [
    "market microstructure",
    "order flow toxicity",
    "price impact high frequency",
    "limit order book deep learning",
    "crypto perpetual futures",
]

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

EXTRACT_SYSTEM = """You are a quantitative researcher extracting testable trading signal ideas from academic papers.

Focus on ideas applicable to crypto perpetual futures at 100ms-5min horizons:
- Order book features predicting short-term returns
- Flow patterns indicating informed trading
- Regime-dependent effects on signal efficacy
- Cross-asset lead-lag relationships
- Novel feature engineering approaches

Return JSON: {"ideas": ["idea 1", "idea 2", ...]}
If no relevant ideas, return {"ideas": []}.
Maximum 3 ideas per paper. Each idea should be specific and testable."""


def fetch_recent_papers(
    queries: list[str] | None = None,
    max_results: int = 20,
    days_back: int = 30,
) -> list[dict]:
    """Query arXiv API for recent papers matching keywords."""
    queries = queries or SEARCH_QUERIES
    papers = []
    seen_ids = set()

    for query in queries:
        encoded = urllib.parse.quote(f'all:"{query}"')
        url = f"{ARXIV_API}?search_query={encoded}&sortBy=submittedDate&sortOrder=descending&max_results={max_results // len(queries)}"

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)

            for entry in root.findall("atom:entry", NS):
                arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
                published = entry.find("atom:published", NS).text

                categories = []
                for cat in entry.findall("arxiv:primary_category", NS):
                    categories.append(cat.get("term", ""))

                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "published": published,
                    "categories": categories,
                })

            # Be polite to arXiv API
            time.sleep(3)

        except Exception as e:
            log.warning("arXiv query '%s' failed: %s", query, e)

    log.info("Fetched %d papers from arXiv", len(papers))
    return papers


def _filter_unseen(papers: list[dict], store) -> list[dict]:
    """Filter out papers already in the database."""
    conn = store._conn
    unseen = []
    for p in papers:
        row = conn.execute(
            "SELECT 1 FROM arxiv_papers WHERE arxiv_id = ?", (p["arxiv_id"],)
        ).fetchone()
        if not row:
            unseen.append(p)
    return unseen


def _store_paper(store, paper: dict, ideas: list[str] | None = None) -> None:
    """Insert or update paper in database."""
    conn = store._conn
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO arxiv_papers "
            "(arxiv_id, title, abstract, published, categories, ideas, "
            " processed, processed_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper["arxiv_id"],
                paper["title"],
                paper["abstract"],
                paper["published"],
                json.dumps(paper.get("categories", [])),
                json.dumps(ideas) if ideas else None,
                1 if ideas is not None else 0,
                now if ideas is not None else None,
                now,
            ),
        )


def extract_ideas(paper: dict, llm) -> list[str]:
    """Use Claude to extract testable signal ideas from a paper abstract."""
    user_msg = f"""Paper: {paper['title']}

Abstract: {paper['abstract'][:2000]}

Extract 0-3 concrete, testable signal ideas for crypto perpetual futures trading."""

    response = llm.call(EXTRACT_SYSTEM, user_msg, tag="arxiv", max_tokens=1024)
    if not response:
        return []

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
        data = json.loads(text.strip())
        return data.get("ideas", [])
    except (json.JSONDecodeError, KeyError) as e:
        log.warning("Failed to parse arXiv extraction response: %s", e)
        return []


def run_arxiv_scan(store, llm, max_papers: int = 5) -> int:
    """Full pipeline: fetch -> filter unseen -> extract ideas -> store."""
    papers = fetch_recent_papers()
    unseen = _filter_unseen(papers, store)

    if not unseen:
        log.info("No new papers to process")
        return 0

    processed = 0
    for paper in unseen[:max_papers]:
        log.info("Processing: %s", paper["title"][:80])

        if llm.budget_remaining > 0:
            ideas = extract_ideas(paper, llm)
            _store_paper(store, paper, ideas)
            if ideas:
                log.info("  Extracted %d ideas", len(ideas))
            processed += 1
        else:
            # Store without processing (will be processed next scan)
            _store_paper(store, paper)
            log.info("  Stored for later (LLM budget exhausted)")

    log.info("Processed %d new papers", processed)
    return processed


def load_recent_ideas(store, max_ideas: int = 10) -> list[str]:
    """Load most recent extracted ideas for use in LLM ideation context."""
    try:
        conn = store._conn
        rows = conn.execute(
            "SELECT ideas FROM arxiv_papers WHERE processed = 1 AND ideas IS NOT NULL "
            "ORDER BY processed_at DESC LIMIT 10"
        ).fetchall()
        all_ideas = []
        for r in rows:
            ideas = json.loads(r["ideas"]) if r["ideas"] else []
            all_ideas.extend(ideas)
        return all_ideas[:max_ideas]
    except Exception:
        return []


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="arXiv Paper Reader")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="Fetch and process recent papers")
    sub.add_parser("list", help="List processed papers")
    sub.add_parser("ideas", help="Show extracted ideas")

    args = parser.parse_args()

    from data.state import StateStore

    store = StateStore(str(ROOT / "data" / "nat.db"))

    if args.command == "scan":
        import tomllib
        with open(ROOT / "config" / "agent.toml", "rb") as f:
            config = tomllib.load(f)
        llm_config = config.get("agent", {}).get("llm", {})
        llm_config["agent_name"] = "arxiv_reader"

        from agent.llm_client import LLMClient
        llm = LLMClient(llm_config, store)

        n = run_arxiv_scan(store, llm)
        print(f"Processed {n} new papers")

    elif args.command == "list":
        conn = store._conn
        rows = conn.execute(
            "SELECT arxiv_id, title, processed FROM arxiv_papers "
            "ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        for r in rows:
            status = "+" if r["processed"] else " "
            print(f"  [{status}] {r['arxiv_id']}: {r['title'][:70]}")

    elif args.command == "ideas":
        ideas = load_recent_ideas(store)
        if ideas:
            for i, idea in enumerate(ideas, 1):
                print(f"  {i}. {idea}")
        else:
            print("  No ideas extracted yet. Run 'scan' first.")

    store.close()


if __name__ == "__main__":
    main()
