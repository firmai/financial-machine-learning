import os
import re
import pandas as pd
from typing import Dict, Any

from conf import PROJECT_ROOT_DIR
from git_util import get_repo_attributes_dict, get_github_client, get_repo_path


def get_repo_list() -> pd.DataFrame:
    """
    Load the repository list from the CSV file.
    Ensures 'repo_path' column is present by extracting from 'url' if missing.
    """
    file_path = os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv')
    repo_df = pd.read_csv(file_path)

    if 'repo_path' not in repo_df.columns:
        repo_df['repo_path'] = repo_df['url'].apply(get_repo_path)

    return repo_df


def get_repo_status() -> None:
    """
    Fetch repository attributes from GitHub and update the local repo list CSV.
    """
    g = get_github_client()
    repo_df = get_repo_list()

    for idx, row in repo_df.iterrows():
        repo_path = row.get('repo_path')
        if pd.notna(repo_path):
            try:
                print(f"Processing [{repo_path}] ...")
                repo = g.get_repo(repo_path)
                repo_attr_dict: Dict[str, Any] = get_repo_attributes_dict(repo)
            except Exception as ex:
                print(f"⚠️ Error processing {repo_path}: {ex}")
                repo_attr_dict = {}

            # Update DataFrame in place
            for k, v in repo_attr_dict.items():
                repo_df.loc[idx, k] = v

    # Save updated repo data
    save_path = os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv')
    repo_df.to_csv(save_path, index=False)
    print(f"✅ Repo status updated and saved to {save_path}")


def parse_readme_md() -> pd.DataFrame:
    """
    Parse repositories from README.md file into a DataFrame.

    Returns:
        pd.DataFrame: Parsed repo info with columns: [name, url, comment, category]

    Usage:
    >>> df = parse_readme_md()
    >>> df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)
    """
    file_path = os.path.join(PROJECT_ROOT_DIR, 'README.md')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"README.md not found at {file_path}")

    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()[11:]  # Skip first 11 heading lines

    all_df_list = []
    line_num = 0

    while line_num < len(lines):
        line = lines[line_num]
        if line.strip().startswith('#'):
            # Found a heading (category)
            heading = line.strip().replace('#', '').strip()

            # Parse lines under this heading
            parsed_list = []
            line_num += 1
            while line_num < len(lines) and not lines[line_num].strip().startswith('#'):
                link_line = lines[line_num].strip()
                if link_line:
                    parsed_set = _parse_readme_line(link_line)
                    if parsed_set:
                        parsed_list.append(parsed_set)
                line_num += 1

            if parsed_list:
                parsed_df = pd.DataFrame(parsed_list, columns=['name', 'url', 'comment'])
                parsed_df['category'] = heading
                all_df_list.append(parsed_df)
        else:
            line_num += 1

    final_df = pd.concat(all_df_list, ignore_index=True) if all_df_list else pd.DataFrame()
    return final_df


def _parse_readme_line(line: str):
    """
    Parse a single README.md repo entry line.
    Expected format: '- [NAME](link) - comment'
    """
    split_sections = line.split('- ')
    if len(split_sections) < 2:
        print(f"⚠️ Unsupported line format: {line}")
        return None

    # Extract optional comment
    comment_str = None if len(split_sections) == 2 else '-'.join(split_sections[2:]).strip()

    # Extract title
    title_and_link = split_sections[1].strip()
    title_match = re.search(r'\[(.*?)\]', title_and_link)
    title_str = title_match.group(1) if title_match else None

    # Extract link
    link_match = re.search(r'\((.*?)\)', title_and_link)
    link_str = link_match.group(1) if link_match else None

    return (title_str, link_str, comment_str)


if __name__ == '__main__':
    get_repo_status()
