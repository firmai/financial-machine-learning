from conf import PROJECT_ROOT_DIR
import os
import pandas as pd


@DeprecationWarning
def generate_wiki_per_category(output_path):
    """

    :param output_path:
    """
    repo_path = os.path.join(PROJECT_ROOT_DIR, 'raw_data', 'url_list.csv')
    repo_df = pd.read_csv(repo_path)
    for category in repo_df['category'].unique():
        category_df = repo_df[repo_df['category'] == category].copy()
        url_md_list = []
        for idx, irow in category_df[['name', 'url']].iterrows():
            url_md_list.append('[{}]({})'.format(irow['name'], irow['url']))
        formatted_df = pd.DataFrame({
            'repo': url_md_list,
            'comment': category_df['comment'],
            'created_at': category_df['created_at'],
            'last_commit': category_df['last_commit'],
            'star_count': category_df['star_count']
        })
        output_path_full = os.path.join(output_path, '{}.md'.format(category))
        with open(output_path_full, 'w') as f:
            f.write(formatted_df.to_markdown(index=False))
