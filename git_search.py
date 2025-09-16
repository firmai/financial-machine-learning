import datetime
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from conf import PROJECT_ROOT_DIR
from git_status import get_repo_list
from git_util import get_github_client, get_repo_attributes_dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def convert_repo_list_to_df(repo_list, category: str) -> pd.DataFrame:
    """Convert a list of GitHub repos to a DataFrame with extra attributes."""
    df_list = [
        {
            **get_repo_attributes_dict(repo),
            "name": repo.name,
            "comment": "NEW",
            "category": category,
            "repo_path": repo.full_name,
            "url": f"https://github.com/{repo.full_name}",
        }
        for repo in repo_list
    ]
    return pd.DataFrame(df_list)


def build_qualifier_dict(min_stars: Optional[int] = None,
                         created_at: Optional[str] = None,
                         pushed_date: Optional[str] = None) -> Dict[str, str]:
    """Helper to build qualifier dictionary for GitHub search."""
    qualifiers = {}
    if min_stars is not None:
        qualifiers["stars"] = f">={min_stars}"
    if created_at:
        qualifiers["created"] = f">={created_at}"
    if pushed_date:
        qualifiers["pushed"] = f">={pushed_date}"
    return qualifiers


def search_repo(search_term: str, qualifier_dict: Dict[str, str]):
    """Search GitHub repositories with term and qualifiers."""
    g = get_github_client()
    qualifier_str = " ".join([f"{k}:{v}" for k, v in qualifier_dict.items()])
    final_search_term = f"{search_term} {qualifier_str}".strip()
    return g.search_repositories(final_search_term)


def search_repo_simple(search_term: str,
                       min_stars: Optional[int] = None,
                       created_at: Optional[str] = None,
                       pushed_date: Optional[str] = None):
    """Search GitHub repos with simplified parameters."""
    qualifiers = build_qualifier_dict(min_stars, created_at, pushed_date)
    return search_repo(search_term or "", qualifiers)


def search_repo_multiple_terms(term_list: List[str],
                               category: str,
                               min_stars: Optional[int] = None,
                               created_at: Optional[str] = None,
                               pushed_date: Optional[str] = None,
                               drop_duplicate: bool = True) -> pd.DataFrame:
    """Search multiple terms for a category and merge results into a DataFrame."""
    repo_df_list = [
        convert_repo_list_to_df(
            search_repo_simple(term, min_stars, created_at, pushed_date),
            category
        )
        for term in term_list
    ]

    combined_df = pd.concat(repo_df_list, ignore_index=True)
    if drop_duplicate:
        combined_df = combined_df.drop_duplicates(subset="repo_path")

    combined_df["finml_added_date"] = datetime.datetime.now()
    return combined_df


def search_new_repo_by_category(category: str,
                                min_stars: int = 100,
                                existing_repo_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Search new repos by predefined category keywords."""
    logging.info(f"Searching for category [{category}] with min stars {min_stars}")

    category_terms = {
        "Deep Learning And Reinforcement Learning": [
            "deep learning trading", "deep learning finance",
            "reinforcement learning trading", "reinforcement learning finance"
        ],
        "Other Models": ["machine learning trading", "machine learning finance"],
        "Data Processing Techniques and Transformations": [
            "data transformation trading", "data transformation finance", "data transformation time series",
            "data processing trading", "data processing finance", "power transform trading", "power transform finance",
            "standardization normalization trading", "standardization normalization finance"
        ],
        "Portfolio Selection and Optimisation": [
            "portfolio optimization machine learning finance", "portfolio optimization machine learning trading",
            "portfolio construction machine learning finance", "portfolio construction machine learning trading",
            "portfolio optimization finance", "portfolio optimization trading",
            "portfolio construction finance", "portfolio construction trading"
        ],
        "Factor and Risk Analysis": [
            "risk factor finance", "risk factor trading", "risk premia factor finance", "risk premia factor trading",
            "style factor finance", "style factor trading", "macro factor finance", "macro factor trading"
        ],
        "Unsupervised": ["unsupervised finance", "unsupervised trading"],
        "Textual": ["NLP finance", "NLP trading"],
        "Derivatives and Hedging": [
            "derivatives finance", "derivatives trading", "quantlib trading", "quantlib finance",
            "hedging finance", "hedging trading", "option trading", "option finance",
            "delta hedge trading", "delta hedge finance"
        ],
        "Fixed Income": [
            "corporate bond finance", "corporate bond trading", "muni bond trading", "muni bond finance",
            "investment grade finance", "investment grade trading", "high yield trading", "high yield finance",
            "credit rating trading", "credit rating finance", "fixed income trading", "fixed income finance",
            "corporate bond", "muni bond", "credit rating"
        ],
        "Alternative Finance": [
            "private equity", "venture capital", "real estate trading", "real estate finance",
            "alternative asset trading", "alternative asset finance", "commodity trading", "commodity finance",
            "farmland finance", "farmland trading"
        ],
        "Extended Research": [
            "fraud detection", "behavioural finance", "corporate finance", "financial economics",
            "mathematical finance", "liquidity finance", "fx trading", "company life cycle",
            "merger and acquisition", "farmland trading", "HFT", "high frequency trading"
        ],
        "Courses": [
            "finance courses", "machine learning courses", "quantitative finance courses",
            "time series courses", "data science courses", "financial engineering courses"
        ],
        "Data": [
            "financial data", "time series data", "company fundamental data", "crypto data", "earnings data",
            "fixed income data", "fx data", "etf data", "finance database", "sec edgar", "economic data",
            "investment data", "fund data", "options data", "financial index data", "futures data",
            "cryptocurrencies data", "money market data"
        ]
    }

    if category not in category_terms:
        logging.warning(f"No search terms defined for category {category}")
        return None

    # Adjust stars requirement for specific categories
    scale_factors = {
        "Data Processing Techniques and Transformations": 0.5,
        "Factor and Risk Analysis": 0.05,
        "Unsupervised": 0.1,
        "Fixed Income": 0.2,
        "Alternative Finance": 0.5,
        "Extended Research": 0.5,
        "Courses": 2,
    }
    effective_stars = int(min_stars * scale_factors.get(category, 1))

    combined_df = search_repo_multiple_terms(
        category_terms[category], category, effective_stars
    )

    # Filter out already existing repos
    if existing_repo_df is not None and not combined_df.empty:
        combined_df = combined_df[
            ~combined_df["repo_path"].str.lower().isin(existing_repo_df["repo_path"].dropna().str.lower())
        ]

    return combined_df if not combined_df.empty else None


def search_new_repo_and_append(min_stars: int = 100, filter_list: Optional[List[str]] = None):
    """Search new repos by category and append to existing repo list."""
    repo_df = get_repo_list()
    category_list = [c for c in repo_df["category"].unique() if filter_list is None or c in filter_list]

    new_repo_list = [df for c in category_list if (df := search_new_repo_by_category(c, min_stars, repo_df)) is not None]

    if not new_repo_list:
        logging.info("No new repos found.")
        return

    new_repo_df = pd.concat(new_repo_list, ignore_index=True).drop_duplicates(subset="repo_path")
    final_df = pd.concat([repo_df, new_repo_df], ignore_index=True).sort_values(by="category")

    output_path = os.path.join(PROJECT_ROOT_DIR, "raw_data", "url_list.csv")
    final_df.to_csv(output_path, index=False)
    logging.info(f"Repo list updated and saved to {output_path}")


def search_new_repo_by_category_per_day(min_stars: int = 100):
    """Rotate categories daily to avoid hitting GitHub API limits."""
    repo_df = get_repo_list()
    category_list = repo_df["category"].unique().tolist()
    n_category = len(category_list)
    n_repo_per_day = -(-n_category // 7)  # ceiling division
    today_idx = datetime.datetime.today().weekday()

    repo_to_process = category_list[today_idx * n_repo_per_day:(today_idx + 1) * n_repo_per_day]
    search_new_repo_and_append(min_stars, filter_list=repo_to_process)


if __name__ == "__main__":
    search_new_repo_by_category_per_day(min_stars=100)
