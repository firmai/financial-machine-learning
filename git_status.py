import os
from conf import PROJECT_ROOT_DIR
import re
import pandas as pd
from github import Github


def get_repo_status():
    github_token = os.environ.get('GITHUB_TOKEN')
    g = Github(github_token)
    repo_df = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'))

    for idx, row in repo_df.iterrows():
        url = row['url']
        if 'https://github.com/' in url:
            print('processing [{}]'.format(url))
            url_query = url.replace('https://github.com/', '')
            url_format = '/'.join(url_query.split('/')[:2])
            try:
                repo = g.get_repo(url_format)
                repo_df.loc[idx, 'last_update'] = repo.updated_at
                repo_df.loc[idx, 'star_count'] = repo.stargazers_count
                repo_df.loc[idx, 'fork_count'] = repo.forks_count
                repo_df.loc[idx, 'contributors_count'] = repo.get_contributors().totalCount
            except Exception as ex:
                print(ex)
                repo_df.loc[idx, 'last_update'] = None
    repo_df.to_csv(os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv'), index=False)


# @DeprecationWarning
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
