import os
from typing import Dict

from conf import PROJECT_ROOT_DIR
import re
import pandas as pd
from github import Github, Repository


# generic search functions
def search_repo(search_term: str, qualifier_dict: Dict):
    github_token = os.environ.get('GIT_TOKEN')
    g = Github(github_token)
    qualifier_str = ' '.join(['{}:{}'.format(k, v) for k, v in iter(qualifier_dict.items())])
    if qualifier_str != '':
        final_search_term = '{} {}'.format(search_term, qualifier_str)
    else:
        final_search_term = search_term
    repo_result = g.search_repositories(final_search_term)
    return repo_result


def search_repo_simple(search_term: str = None,
                       min_stars_number: int = None,
                       created_at: str = None,
                       pushed_date: str = None
                       ):
    """

    :param search_term:
    :param min_stars_number:
    :param created_at:
    :param pushed_date:
    usage:
    >>> search_term = '(deep learning) AND trading'
    >>> min_stars_number = 100
    >>> created_at = None
    >>> pushed_date = None
    """
    if search_term is None:
        _search_term = ''
    else:
        _search_term = search_term

    qualifier_dict = {}
    if min_stars_number is not None:
        qualifier_dict['stars'] = '>={}'.format(min_stars_number)

    if created_at is not None:
        qualifier_dict['created'] = '>={}'.format(created_at)

    if pushed_date is not None:
        qualifier_dict['pushed'] = '>={}'.format(pushed_date)
    search_result = search_repo(_search_term, qualifier_dict)
    return search_result


# *******
# topic specific search functions
# *******
def convert_repo_list_to_df(repo_list, category):
    df_list = []
    for repo in repo_list:
        attr_dict = get_repo_attributes_dict(repo)
        attr_dict['name'] = repo.name
        attr_dict['comment'] = 'NEW'
        attr_dict['category'] = category
        attr_dict['repo_path'] = repo.full_name
        attr_dict['url'] = 'https://github.com/{}'.format(repo.full_name)
        df_list.append(attr_dict)
    result_df = pd.DataFrame(df_list)
    return result_df


def search_new_repo_and_append(min_stars_number: int = 100):
    repo_df = get_repo_list()
    category_list = repo_df['category'].unique().tolist()
    new_repo_list = []
    for category in category_list:
        if category == 'Deep Learning':
            # github not yet support OR operator, issue here
            # https://github.com/isaacs/github/issues/660
            # hence run the search terms twice and combine
            search_term = 'deep learning trading'
            repo_list = search_repo_simple(search_term, min_stars_number)
            top_df = convert_repo_list_to_df(repo_list, category)
            search_term = 'deep learning finance'
            repo_list = search_repo_simple(search_term, min_stars_number)
            bottom_df = convert_repo_list_to_df(repo_list, category)
            combined_df = pd.concat([top_df, bottom_df]).reset_index(drop=True)

            # only find ones that need to be inserted
            combined_df = combined_df[~combined_df['repo_path'].str.lower().isin(repo_df['repo_path'].str.lower())]
            new_repo_list.append(combined_df)
    new_repo_df = pd.concat(new_repo_list).reset_index(drop=True)
    final_df = pd.concat([repo_df.drop('repo_path', axis=1), new_repo_df.drop('repo_path', axis=1)]).reset_index(
        drop=True)
    final_df = final_df.sort_values(by='category')
    final_df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)


# *******
# saved repo list, treat it as database for now
# *******
def get_repo_list():
    repo_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'))
    repo_df['repo_path'] = repo_df['url'].apply(get_repo_path)
    return repo_df


# *******
# repo specific information
# *******
def get_repo_path(in_url):
    repo_path = None
    if 'https://github.com/' in in_url:
        url_query = in_url.replace('https://github.com/', '')
        repo_path = '/'.join(url_query.split('/')[:2])
    return repo_path


def get_last_commit_date(input_repo: Repository):
    """
    get latest commit from repo
    :param input_repo:
    :return:
    """
    page = input_repo.get_commits().get_page(0)[0]
    return page.commit.author.date


def get_repo_attributes_dict(input_repo: Repository):
    result_dict = {
        'created_at': input_repo.created_at,
        'last_commit': get_last_commit_date(input_repo),
        'last_update': input_repo.updated_at,
        'star_count': input_repo.stargazers_count,
        'fork_count': input_repo.forks_count,
        'contributors_count': input_repo.get_contributors().totalCount

    }
    return result_dict


def get_repo_status():
    github_token = os.environ.get('GIT_TOKEN')
    g = Github(github_token)
    repo_df = get_repo_list()
    for idx, row in repo_df.iterrows():
        repo_path = row['repo_path']
        if repo_path is not None:
            try:
                print('processing [{}]'.format(repo_path))
                repo = g.get_repo(repo_path)
                repo_attr_dict = get_repo_attributes_dict(repo)
            except Exception as ex:
                print(ex)
                repo_attr_dict = {}

            for k, v in iter(repo_attr_dict.items()):
                repo_df.loc[idx, k] = v
    repo_df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)


@DeprecationWarning
def parse_readme_md():
    """

    :return:
    usage:
    >>> df = parse_readme_md()
    >>> df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)
    """
    file_path = os.path.join(PROJECT_ROOT_DIR, 'README.md')
    with open(file_path) as f:
        lines = f.readlines()[11:]  # skip heading
        all_df_list = []
        for line_num in range(len(lines)):
            line = lines[line_num]
            if line.strip().startswith('#'):
                # find a heading
                heading = line.strip().replace('#', '').replace('\n', '').strip()
                # parse until next # or eof
                parsed_list = []
                line_num += 1
                while line_num < len(lines) and not lines[line_num].strip().startswith('#'):
                    link_line = lines[line_num].replace('\n', '').strip()
                    if len(link_line) > 0:
                        # usually in the format of '- [NAME](link) - comment
                        split_sections = link_line.split('- ')
                        if len(split_sections) == 2:
                            comment_str = None
                        elif len(split_sections) >= 3:
                            comment_str = '-'.join(split_sections[2:]).strip()
                        else:
                            raise Exception('link_line [{}] not supported'.format(link_line))

                        title_and_link = split_sections[1].strip()
                        title = re.search(r'\[(.*?)\]', title_and_link)
                        title_str = None
                        if title is not None:
                            title_str = title.group(1)
                            title_and_link = title_and_link.replace('[{}]'.format(title_str), '')
                        m_link = re.search(r'\((.*?)\)', title_and_link)
                        link_str = None
                        if m_link is not None:
                            link_str = m_link.group(1)
                        parsed_set = (title_str, link_str, comment_str)
                        parsed_list.append(parsed_set)
                    line_num += 1
                parsed_df = pd.DataFrame(parsed_list, columns=['name', 'url', 'comment'])
                parsed_df['category'] = heading
                all_df_list.append(parsed_df)
    final_df = pd.concat(all_df_list).reset_index(drop=True)
    return final_df


if __name__ == '__main__':
    get_repo_status()
    search_new_repo_and_append(min_stars_number=100)
