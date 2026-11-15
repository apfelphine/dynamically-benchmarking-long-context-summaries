import datetime
import logging
import os
from typing import List, Optional

import requests
import tqdm
import wikipediaapi

logging.getLogger("wikipediaapi").setLevel(logging.WARNING)


class WikipediaCrawler:
    def __init__(
        self,
        language: str = "en",
        timeout: int = 20,
    ):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=os.environ["WIKI_USERAGENT"],
            language=language,
            extract_format=wikipediaapi.ExtractFormat.HTML,
            timeout=timeout,
        )


    def get_last_revision_before(
        self, title: str, target_date: datetime.datetime
    ):
        session = requests.Session()
        url = f"https://{self.wiki.language}.wikipedia.org/w/api.php"

        # Format target_date to MediaWiki API timestamp format (ISO 8601)
        formatted_date = target_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "revisions",
            "rvprop": "timestamp|ids|user",
            "rvlimit": 1,  # We only need the single closest revision
            "rvstart": formatted_date,  # Start looking backward from this date
            "rvdir": "older",  # Go backwards in time from the start date
        }

        headers = {"User-Agent": os.environ["WIKI_USERAGENT"]}

        response = session.get(url, params=params, headers=headers)
        data = response.json()

        pages = data.get("query", {}).get("pages", [])

        if not pages:
            return None

        # Handle both list and dict response formats from MediaWiki API
        page_data = pages[0] if isinstance(pages, list) else list(pages.values())[0]


        if "missing" in page_data:
            raise ValueError(f"The page '{title}' does not exist on {self.wiki.language}.wikipedia.")

        revisions = page_data.get("revisions", [])
        if not revisions:
            return None  # No revisions found before this date (page might be newer)

        latest_rev = revisions[0]
        return latest_rev.get("timestamp")

    def get_pages(self, page_names: List[str]) -> List[wikipediaapi.WikipediaPage]:
        """Fetch a list of page objects."""
        pages = []
        for name in tqdm.tqdm(page_names, desc="Downloading pages"):
            page = self.wiki.page(name)
            if page.namespace != wikipediaapi.Namespace.MAIN:
                continue
            page._fetch("info")  # noqa
            pages.append(page)
        return pages

    def get_linked_pages(
        self,
        page_name: str,
        allowed_category_names: Optional[List[str]] = None,
    ) -> List[wikipediaapi.WikipediaPage]:
        """
        Find pages linked from a specific page, filtering by specific criteria.
        """
        page = self.wiki.page(page_name)

        # Convert lists to sets for O(1) lookups
        allowed_cats_set = (
            set(allowed_category_names) if allowed_category_names else None
        )

        found_pages = []

        # Iterate over links on the page
        desc = f"Scanning linked pages on '{page_name}'"
        for link_title, link_page in tqdm.tqdm(page.links.items(), desc=desc):
            if allowed_cats_set:
                link_page._fetch("info")  # noqa

            # Filter by Category
            if allowed_cats_set is not None:
                page_cats = link_page.categories.keys()
                if not any(
                    any(allowed in cat_key for allowed in allowed_cats_set)
                    for cat_key in page_cats
                ):
                    continue

            found_pages.append(link_page)

        return found_pages

    def find_category_pages(
        self, category_name: str, max_level: int = 0, exclude_main: bool = False
    ) -> List[wikipediaapi.WikipediaPage]:
        """
        Recursively find all pages within a category.
        """
        cat_page = self.wiki.page(category_name)

        if not cat_page.exists():
            raise Exception(f"Category '{category_name}' does not exist.")

        pages = self._recurse_category_members(
            cat_page.categorymembers, level=0, max_level=max_level
        )

        if exclude_main:
            # Clean string to match title format (e.g. "Category:Physics" -> "Physics")
            clean_name = category_name.replace("_", " ").replace("Category:", "")
            pages = [p for p in pages if p.title != clean_name]

        return pages

    def _recurse_category_members(
        self, categorymembers, level, max_level
    ) -> List[wikipediaapi.WikipediaPage]:
        """Helper method for recursive category crawling."""
        res = []
        for c in categorymembers.values():
            if c.ns == wikipediaapi.Namespace.MAIN:
                res.append(c)

            if c.ns == wikipediaapi.Namespace.CATEGORY and (
                max_level is None or level < max_level
            ):
                res.extend(
                    self._recurse_category_members(
                        c.categorymembers, level=level + 1, max_level=max_level
                    )
                )
        return res
