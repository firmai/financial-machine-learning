from github import Github, Repository, GithubException
import os
import datetime


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
    result_dict = {}
    try:
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
    except Exception as e:
        print(e)

    return result_dict
