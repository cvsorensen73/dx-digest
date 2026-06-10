#!/usr/bin/env python3
"""DX Digest — per-customer AI-curated news dashboard for Technical Account Managers."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests
import anthropic
import yaml
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config & customers ────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_customers(path: str = "customers.md") -> list[dict]:
    """Parse customers.md into a list of customer dicts."""
    text = Path(path).read_text(encoding="utf-8")
    customers = []
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                customers.append(current)
            current = {"name": line[3:].strip(), "industry": "", "region": "", "notes": ""}
        elif current:
            m = re.match(r"-\s+\*\*Industry:\*\*\s+(.+)", line)
            if m:
                current["industry"] = m.group(1).strip()
            m = re.match(r"-\s+\*\*Region:\*\*\s+(.+)", line)
            if m:
                current["region"] = m.group(1).strip()
            m = re.match(r"-\s+\*\*Notes:\*\*\s+(.+)", line)
            if m:
                current["notes"] = m.group(1).strip()
    if current:
        customers.append(current)
    log.info("Loaded %d customers from %s", len(customers), path)
    return customers


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_rss(feed_url: str, max_items: int = 30) -> list[dict]:
    try:
        feed = feedparser.parse(feed_url)
        articles = []
        for entry in feed.entries[:max_items]:
            articles.append({
                "title": (entry.get("title") or "").strip(),
                "url": entry.get("link", ""),
                "source": (feed.feed.get("title") or feed_url).strip()[:80],
                "published": entry.get("published", ""),
                "snippet": re.sub(r"<[^>]+>", "", entry.get("summary") or "")[:500],
            })
        log.info("RSS %s → %d articles", feed_url[:70], len(articles))
        return articles
    except Exception as exc:
        log.warning("RSS fetch failed (%s): %s", feed_url[:70], exc)
        return []


def fetch_newsapi(queries: list[str], api_key: str) -> list[dict]:
    articles = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "language": "en", "sortBy": "publishedAt",
                        "pageSize": 20, "apiKey": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            for a in resp.json().get("articles", []):
                url = a.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    articles.append({
                        "title": (a.get("title") or "").strip(),
                        "url": url,
                        "source": a.get("source", {}).get("name", "")[:80],
                        "published": a.get("publishedAt", ""),
                        "snippet": (a.get("description") or "")[:500],
                    })
        except Exception as exc:
            log.warning("NewsAPI failed for '%s': %s", query, exc)
    log.info("NewsAPI → %d articles total", len(articles))
    return articles


def build_customer_rss_feeds(customer: dict) -> list[str]:
    """Generate Google News RSS URLs for a specific customer."""
    name = customer["name"]
    industry = customer["industry"].split("/")[0].strip()
    region = customer["region"].split(",")[0].strip()
    queries = [
        f"{name} digital banking technology",
        f"{name} strategy innovation",
        f"{industry} {region} digital transformation",
        f"{industry} AI customer experience 2025",
        f"{region} {industry} regulation technology",
    ]
    base = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
    return [base + quote_plus(q) for q in queries]


def deduplicate(articles: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for a in articles:
        key = a["url"].rstrip("/")
        if key and key not in seen and a["title"]:
            seen.add(key)
            unique.append(a)
    return unique


# ── AI processing ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You assist a Technical Account Manager (TAM) at a technology company. \
The TAM manages relationships with financial services customers across the Nordic region. \
Your job is to identify the most relevant news articles for each customer and \
suggest concrete talking points the TAM can use when meeting with that customer.\
"""


def process_customer(
    customer: dict,
    articles: list[dict],
    client: anthropic.Anthropic,
    config: dict,
) -> dict:
    """Score articles for relevance to this customer and generate TAM actions."""
    if not articles:
        return {**customer, "articles": [], "actions": []}

    min_score = config.get("min_relevance_score", 6)
    nordic_bonus = config.get("nordic_bonus", 1)
    limit = config.get("max_articles_per_customer", 12)

    numbered = "\n".join(
        f'{i}. TITLE: {a["title"]}\n   SOURCE: {a["source"]}\n   SNIPPET: {a["snippet"][:300]}'
        for i, a in enumerate(articles)
    )

    prompt = f"""\
Customer: {customer["name"]}
Industry: {customer["industry"]}
Region: {customer["region"]}
Context: {customer.get("notes", "")}

Score each of the {len(articles)} articles below for relevance (0–10) to this customer.
Apply a +{nordic_bonus} bonus for articles that are specifically relevant to the Nordic/Scandinavian market.
Write a 2-sentence abstract (plain text) for articles with a final score >= {min_score}.

Then, based on the most relevant articles, suggest exactly 3 specific talking points for the TAM \
to raise with {customer["name"]}. Each talking point should be 1–2 sentences, concrete, and \
actionable — something the TAM can say in a meeting to add value.

Articles:
{numbered}

Reply with JSON only — no other text:
{{
  "articles": [{{"index": 0, "score": 8, "abstract": "..."}}],
  "actions": ["Talking point 1.", "Talking point 2.", "Talking point 3."]
}}
Only include articles with final score >= {min_score}.\
"""

    try:
        response = client.messages.create(
            model=config.get("ai_model", "claude-sonnet-4-6"),
            max_tokens=4096,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            log.error("No JSON in AI response for %s", customer["name"])
            return {**customer, "articles": [], "actions": []}

        data = json.loads(match.group())
        scored = []
        for r in data.get("articles", []):
            idx = r.get("index", -1)
            if 0 <= idx < len(articles):
                article = dict(articles[idx])
                article["score"] = r.get("score", 0)
                article["abstract"] = r.get("abstract", article["snippet"][:200])
                # stable id for localStorage keying
                article["id"] = _url_id(article["url"])
                scored.append(article)

        scored.sort(key=lambda x: x["score"], reverse=True)
        log.info("%s → %d relevant articles", customer["name"], len(scored))

        return {
            **customer,
            "articles": scored[:limit],
            "actions": data.get("actions", []),
        }

    except Exception as exc:
        log.error("AI processing failed for %s: %s", customer["name"], exc)
        return {**customer, "articles": [], "actions": []}


def generate_summary(customers: list[dict], client: anthropic.Anthropic, config: dict) -> str:
    """Generate a cross-customer executive summary paragraph."""
    sections = []
    for c in customers:
        if c["articles"]:
            titles = "; ".join(a["title"] for a in c["articles"][:5])
            sections.append(f"{c['name']} ({c['industry']}): {titles}")

    if not sections:
        return "No relevant articles found this week."

    prompt = f"""\
Below are this week's top news topics per customer in a TAM's portfolio. \
Write a 3–4 sentence executive summary of the most important themes and trends \
cutting across these customers. Focus on what matters strategically for a TAM \
advising Nordic financial services firms on digital transformation. Plain text only.

{chr(10).join(sections)}\
"""
    try:
        response = client.messages.create(
            model=config.get("ai_model", "claude-sonnet-4-6"),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as exc:
        log.error("Summary generation failed: %s", exc)
        return ""


def _url_id(url: str) -> str:
    """Short stable ID derived from URL for use in localStorage keys."""
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ── HTML render ───────────────────────────────────────────────────────────────

def render_html(customers: list[dict], summary: str, output_path: str, config: dict) -> None:
    with open("template.html") as f:
        template = Template(f.read())

    html = template.render(
        customers=customers,
        summary=summary,
        generated_at=datetime.now(timezone.utc).strftime("%B %d, %Y — %H:%M UTC"),
        week_label=datetime.now(timezone.utc).strftime("Week %V · %Y"),
        total_articles=sum(len(c["articles"]) for c in customers),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Rendered → %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    customers = load_customers()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic()

    # Gather articles: global feeds
    raw: list[dict] = []
    for url in config.get("rss_feeds", []):
        raw.extend(fetch_rss(url))

    newsapi_key = os.environ.get("NEWSAPI_KEY")
    if newsapi_key:
        raw.extend(fetch_newsapi(config.get("newsapi_queries", []), newsapi_key))

    # Gather articles: per-customer feeds
    for customer in customers:
        for url in build_customer_rss_feeds(customer):
            raw.extend(fetch_rss(url))

    raw = deduplicate(raw)
    log.info("Total unique articles to score: %d", len(raw))

    # Process each customer
    results = []
    for customer in customers:
        log.info("Processing customer: %s", customer["name"])
        result = process_customer(customer, raw, client, config)
        results.append(result)

    summary = generate_summary(results, client, config)
    render_html(results, summary, "docs/index.html", config)

    total = sum(len(c["articles"]) for c in results)
    print(f"\nDone. {total} articles across {len(results)} customers → docs/index.html")


if __name__ == "__main__":
    main()
