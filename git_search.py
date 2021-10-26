import datetime
import os
from typing import Dict, List

import pandas as pd

from conf import PROJECT_ROOT_DIR
from git_status import get_repo_list
from git_util import get_github_client, get_repo_attributes_dict


def convert_repo_list_to_df(repo_list, category):
    df_list = []
    for repo in repo_list:
        print(repo)
        attr_dict = get_repo_attributes_dict(repo)
        attr_dict['name'] = repo.name
        attr_dict['comment'] = 'NEW'
        attr_dict['category'] = category
        attr_dict['repo_path'] = repo.full_name
        attr_dict['url'] = 'https://github.com/{}'.format(repo.full_name)
        df_list.append(attr_dict)
    result_df = pd.DataFrame(df_list)
    return result_df


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


def search_new_repo_by_category(category: str,
                                min_stars_number: int = 100,
                                existing_repo_df: pd.DataFrame = None):
    """

    :param category:
    :param min_stars_number:
    :param existing_repo_df:
    :return:
    usage:
    >>> category = 'Data Processing Techniques and Transformations'
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
                                                  'data transformation time series',
                                                  'data processing trading',
                                                  'data processing finance',
                                                  'power transform trading',
                                                  'power transform finance',
                                                  'standardization normalization trading',
                                                  'standardization normalization finance',
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
                                                 min_stars_number=min_stars_number
                                                 )
    elif category == 'Factor and Risk Analysis':
        combined_df = search_repo_multiple_terms(['risk factor finance',
                                                  'risk factor trading',
                                                  'risk premia factor finance',
                                                  'risk premia factor trading',
                                                  'style factor finance',
                                                  'style factor trading',
                                                  'macro factor finance',
                                                  'macro factor trading',
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.05)
                                                 )
    elif category == 'Unsupervised':
        combined_df = search_repo_multiple_terms(['unsupervised finance',
                                                  'unsupervised trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.1)
                                                 )
    elif category == 'Textual':
        combined_df = search_repo_multiple_terms(['NLP finance',
                                                  'NLP trading'
                                                  ],
                                                 category,
                                                 min_stars_number=min_stars_number
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
                                                 min_stars_number=min_stars_number
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
                                                  'fixed income finance',
                                                  'corporate bond',
                                                  'muni bond',
                                                  'credit rating'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.2)
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
    elif category == 'Extended Research':
        combined_df = search_repo_multiple_terms(['fraud detection',
                                                  'behavioural finance',
                                                  'corporate finance',
                                                  'financial economics',
                                                  'mathematical finance',
                                                  'liquidity finance',
                                                  'fx trading',
                                                  'company life cycle',
                                                  'merger and acquisition',
                                                  'farmland trading',
                                                  'HFT',
                                                  'high frequency trading'
                                                  ],
                                                 category,
                                                 min_stars_number=int(min_stars_number * 0.5)
                                                 )
    elif category == 'Courses':
        combined_df = search_repo_multiple_terms(['finance courses',
                                                  'machine learning courses',
                                                  'quantitative finance courses',
                                                  'time series courses',
                                                  'data science courses',
                                                  'financial engineering courses'
                                                  ],
                                                 category,
                                                 min_stars_number=min_stars_number * 2
                                                 )
    elif category == 'Data':
        combined_df = search_repo_multiple_terms(['financial data',
                                                  'time series data',
                                                  'company fundamental data',
                                                  'crypto data',
                                                  'earnings data',
                                                  'fixed income data',
                                                  'fx data',
                                                  'etf data',
                                                  'finance database',
                                                  'sec edgar',
                                                  'economic data',
                                                  'investment data',
                                                  'fund data',
                                                  'options data',
                                                  'financial index data',
                                                  'futures data',
                                                  'cryptocurrencies data',
                                                  'money market data'
                                                  ],
                                                 category,
                                                 min_stars_number=min_stars_number
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


def search_new_repo_by_category_per_day(min_stars_number: int = 100):
    repo_df = get_repo_list()
    category_list = repo_df['category'].unique().tolist()
    # based on today's date, pick which category to search to get around api limit
    current_date = datetime.datetime.today()
    n_category = len(category_list)
    days_in_week = 7
    if n_category % days_in_week == 0:
        n_repo_to_process_per_day = int(n_category / days_in_week)
    else:
        n_repo_to_process_per_day = int(n_category / days_in_week) + 1
    today_selection = current_date.weekday()
    repo_to_process = category_list[
                      today_selection * n_repo_to_process_per_day:(today_selection + 1) * n_repo_to_process_per_day]

    search_new_repo_and_append(min_stars_number=min_stars_number, filter_list=repo_to_process)


if __name__ == '__main__':
    search_new_repo_by_category_per_day(min_stars_number=100)
