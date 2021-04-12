import os
from typing import Dict, List
import datetime
from conf import PROJECT_ROOT_DIR
import re
import pandas as pd
from github import Github, Repository


def get_github_client():
    # search for app_client and client secrets first, since this allow higher api request limit
    github_app = os.environ.get('GIT_APP_ID')
    if github_app is None:
        github_token = os.environ.get('GIT_TOKEN')
        g = Github(github_token)
    else:
        github_app_secret = os.environ.get('GIT_APP_SECRET')
        g = Github(
            client_id=github_app,
            client_secret=github_app_secret)
    return g


# generic search functions
def search_repo(search_term: str, qualifier_dict: Dict):
    g = get_github_client()
    qualifier_str = ' '.join(['{}:{}'.format(k, v) for k, v in iter(qualifier_dict.items())])
    if qualifier_str != '':
        final_search_term = '{} {}'.format(search_term, qualifier_str)
    else:
        final_search_term = search_term
    repo_result = g.search_repositories(final_search_term)
    return repo_result


def search_repo_multiple_terms(term_list: List[str],
                               category: str,
                               min_stars_number: int = None,
                               created_at: str = None,
                               pushed_date: str = None,
                               drop_duplicate: bool = True
                               ):
    """

    :param term_list:
    :param category:
    :param min_stars_number:
    :param created_at:
    :param pushed_date:
    :param drop_duplicate:
    :return:
    usage:
    >>> term_list = ['deep learning trading', 'deep learning finance', 'reinforcement learning trading',
    'reinforcement learning finance']
    >>> category = 'Deep Learning And Reinforcement Learning'
    >>> min_stars_number = 100
    >>> created_at = None
    >>> pushed_date = None
    >>> drop_duplicate = True
    """
    repo_df_list = []
    for search_term in term_list:
        repo_list = search_repo_simple(search_term, min_stars_number, created_at=created_at, pushed_date=pushed_date)
        repo_df = convert_repo_list_to_df(repo_list, category)
        repo_df_list.append(repo_df)
    combined_df = pd.concat(repo_df_list).reset_index(drop=True)
    if drop_duplicate:
        combined_df = combined_df.drop_duplicates()
    combined_df['finml_added_date'] = datetime.datetime.now()
    return combined_df


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
    >>> search_term = 'machine learning trading'
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


def search_new_repo_by_category(category: str,
                                min_stars_number: int = 100,
                                existing_repo_df: pd.DataFrame = None):
    """

    :param category:
    :param min_stars_number:
    :param existing_repo_df:
    :return:
    usage:
    >>> category = 'Other Models'
    >>> min_stars_number = 100
    >>> existing_repo_df = get_repo_list()
    """
    print('*** searching for category [{}] ***'.format(category))
    combined_df = None
    if category == 'Deep Learning And Reinforcement Learning':
        combined_df = search_repo_multiple_terms(['deep learning trading',
                                                  'deep learning finance',
                                                  'reinforcement learning trading',
                                                  'reinforcement learning finance'],
                                                 category,
                                                 min_stars_number=min_stars_number
                                                 )

    elif category == 'Other Models':
        combined_df = search_repo_multiple_terms(['machine learning trading',
                                                  'machine learning finance'],
                                                 category,
                                                 min_stars_number=min_stars_number
                                                 )
    elif category == 'Data Processing Techniques and Transformations':
        combined_df = search_repo_multiple_terms(['data transformation trading',
                                                  'data transformation finance',
                                                  'data transformation time series'
                                                  'data processing trading',
                                                  'data processing finance'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Portfolio Selection and Optimisation':
        combined_df = search_repo_multiple_terms(['portfolio optimization machine learning finance',
                                                  'portfolio optimization machine learning trading',
                                                  'portfolio construction machine learning finance',
                                                  'portfolio construction machine learning trading',
                                                  'portfolio optimization finance',
                                                  'portfolio optimization trading',
                                                  'portfolio construction finance',
                                                  'portfolio construction trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Factor and Risk Analysis':
        combined_df = search_repo_multiple_terms(['risk factor finance',
                                                  'risk factor trading',
                                                  'risk premia factor finance',
                                                  'risk premia factor trading',
                                                  'style factor finance',
                                                  'style factor trading',
                                                  'macro factor finance',
                                                  'macro factor trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Unsupervised':
        combined_df = search_repo_multiple_terms(['unsupervised learning finance',
                                                  'unsupervised learning trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Textual':
        combined_df = search_repo_multiple_terms(['NLP finance',
                                                  'NLP trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number)
                                                 )
    elif category == 'Derivatives and Hedging':
        combined_df = search_repo_multiple_terms(['derivatives finance',
                                                  'derivatives trading',
                                                  'quantlib trading',
                                                  'quantlib finance',
                                                  'hedging finance',
                                                  'hedging trading',
                                                  'option trading',
                                                  'option finance',
                                                  'delta hedge trading',
                                                  'delta hedge finance'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Fixed Income':
        combined_df = search_repo_multiple_terms(['corporate bond finance',
                                                  'corporate bond trading',
                                                  'muni bond trading',
                                                  'muni bond finance',
                                                  'investment grade finance',
                                                  'investment grade trading',
                                                  'high yield trading',
                                                  'high yield finance',
                                                  'credit rating trading',
                                                  'credit rating finance',
                                                  'fixed income trading',
                                                  'fixed income finance'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Alternative Finance':
        # don't include crypto here as it will skew the results, consider putting it as a seperate category
        combined_df = search_repo_multiple_terms(['private equity',
                                                  'venture capital',
                                                  'real estate trading',
                                                  'real estate finance',
                                                  'alternative asset trading',
                                                  'alternative asset finance',
                                                  'commodity trading',
                                                  'commodity finance',
                                                  'farmland finance',
                                                  'farmland trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )

    # only find ones that need to be inserted
    if combined_df is not None and not combined_df.empty and existing_repo_df is not None:
        combined_df = combined_df[
            ~combined_df['repo_path'].str.lower().isin(existing_repo_df['repo_path'].dropna().str.lower())]

    return combined_df


def search_new_repo_and_append(min_stars_number: int = 100, filter_list=None):
    """

    :param min_stars_number:
    :param filter_list:
    """
    repo_df = get_repo_list()
    category_list = repo_df['category'].unique().tolist()
    if filter_list is not None:
        category_list = [x for x in category_list if x in filter_list]

    new_repo_list = []
    for category in category_list:
        combined_df = search_new_repo_by_category(category, min_stars_number, repo_df)
        if combined_df is not None:
            new_repo_list.append(combined_df)
    new_repo_df = pd.concat(new_repo_list).reset_index(drop=True)
    # drop duplicate regardless of the category, keep first one for now
    new_repo_df = new_repo_df.drop_duplicates(subset='repo_path')

    final_df = pd.concat([repo_df, new_repo_df]).reset_index(drop=True)

    final_df = final_df.sort_values(by='category')
    final_df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)


# *******
# saved repo list, treat it as database for now
# *******
def get_repo_list():
    repo_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'))
    if 'repo_path' not in repo_df.columns:
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


def get_repo_attributes_dict(input_repo: Repository, last_commit_within_years: int = 2):
    result_dict = {
        'repo_path': input_repo.full_name,
        'created_at': input_repo.created_at,
        'last_commit': get_last_commit_date(input_repo),
        'last_update': input_repo.updated_at,
        'star_count': input_repo.stargazers_count,
        'fork_count': input_repo.forks_count,
        'contributors_count': input_repo.get_contributors().totalCount

    }
    today = datetime.datetime.today()
    check_start_date = datetime.datetime(today.year - last_commit_within_years,
                                         today.month,
                                         today.day)

    if result_dict['last_commit'] >= check_start_date:
        repo_status = 'active'
    else:
        repo_status = 'inactive'
    result_dict['repo_status'] = repo_status

    return result_dict


def get_repo_status():
    g = get_github_client()
    repo_df = get_repo_list()
    for idx, row in repo_df.iterrows():
        repo_path = row['repo_path']
        if not pd.isna(repo_path):
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
