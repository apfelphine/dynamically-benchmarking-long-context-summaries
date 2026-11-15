import asyncio
import os
import re
import time
import unicodedata
from typing import List, Callable, Any

import tiktoken
import tqdm

from core.wikipedia.wiki_crawler import WikipediaCrawler
from core.wikipedia.wiki_parser import WikipediaParser
from experiments.configuration import base_data_dir
from models.corpus import Document, TextStats
from utils.async_json import json_dump

enc = tiktoken.get_encoding("cl100k_base")

TOPIC_CONFIG: dict[str, Callable[[WikipediaCrawler], List[Any]]] = {
    "Universities": lambda c: c.get_links(
        "QS_World_University_Rankings",
        [
            "Universities and colleges established in",
            "Educational institutions established in",
        ],
    ),
    "Neural-Network-Architectures": lambda c: [
        p
        for p in c.find_category_pages("Category:Neural_network_architectures")
        if not p.title.startswith("Types of")
    ],
    "AI-Regulations": lambda c: [
        p
        for p in c.find_category_pages(
            "Category:Regulation_of_artificial_intelligence", exclude_main=True
        )
        + c.find_category_pages("Category:AI_safety", exclude_main=True)
        + c.find_category_pages(
            "Category:Existential_risk_from_artificial_intelligence",
            exclude_main=True,
        )
        if not p.title
        in [
            "Regulation of artificial intelligence",
            "AI Safety",
        ]  # Exclude main articles
    ],
    "AI-Fiction": lambda c: c.get_links(
        "List_of_artificial_intelligence_films",
        ["Films about artificial intelligence"],
    )
    + c.get_links("Artificial_intelligence_in_fiction", ["Novels"]),
    "AI-Philosophy": lambda c: c.find_category_pages(
        "Category:Philosophy_of_artificial_intelligence", exclude_main=True
    ),
    "Education-Systems": lambda c: [
        p
        for p in c.get_links(
            "List_of_secondary_education_systems_by_country", ["Education in"]
        )
        if p.title.startswith("Education in")
    ],
    "Comp-Sci-Pioneers": lambda c: c.get_links(
        "List_of_pioneers_in_computer_science", ["births"]
    ),
    "Scholarships": lambda c: [
        p
        for p in c.find_category_pages("Category:Scholarships_by_country", max_level=1)
        if not p.title
        in [
            "Higher Education Students' Financing Board",
            "Scholarships in the United States",
        ]
        and not p.title.startswith("List")
        and not p.title.startswith("Scholarships")
    ],
}


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


async def main():
    available_topics = TOPIC_CONFIG.keys()
    wiki_crawler = WikipediaCrawler()
    parser = WikipediaParser()
    os.makedirs(base_data_dir, exist_ok=True)

    for topic in available_topics:
        if topic not in TOPIC_CONFIG:
            print(f"Skipping unknown topic: {topic}")
            continue

        print(f"Get data for {topic}...")
        fetcher = TOPIC_CONFIG[topic]
        pages = fetcher(wiki_crawler)
        print("Finished getting pages...")

        description = "Processing pages..."
        pbar = tqdm.tqdm(total=len(pages), desc=description)
        skipped_docs = []

        for p in pages:
            pbar.set_description(f"{description} ({p.title})")
            document, summary, infobox = parser.process_document(p.title)

            document = document.strip()
            summary = summary.strip()

            doc_without_title = document.replace(f"# {p.title}", "").strip()

            # Filter: Content must exist and be significantly longer than summary
            if len(doc_without_title) == 0 or len(doc_without_title) < (
                len(summary) * 1.5
            ):
                skipped_docs.append(p)
                pbar.update(1)
                continue

            relative_path_str = os.path.join(topic, slugify(p.title))

            full_path = os.path.join(base_data_dir, relative_path_str)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            metadata = Document(
                title=p.title,
                url=p.fullurl,
                id=p.pageid,
                summary=TextStats(
                    chars=len(summary),
                    tokens=len(enc.encode(summary)),
                ),
                document=TextStats(
                    chars=len(document),
                    tokens=len(enc.encode(document)),
                ),
                path=relative_path_str,
            )

            # Save metadata
            with open(
                os.path.join(full_path, "document.md"), "w", encoding="utf-8"
            ) as f:
                f.write(document)
            with open(
                os.path.join(full_path, "summary.md"), "w", encoding="utf-8"
            ) as f:
                f.write(summary)
            await json_dump(
                os.path.join(full_path, "metadata.json"), metadata.model_dump()
            )
            pbar.update(1)

        if skipped_docs:
            print(f"\nSkipped {len(skipped_docs)} pages...")
            for d in skipped_docs:
                print(" - " + d.title)
        pbar.close()

    with open(os.path.join(base_data_dir, "info"), "w") as f:
        f.write("Downloaded at: " + time.asctime())


if __name__ == "__main__":
    asyncio.run(main())
