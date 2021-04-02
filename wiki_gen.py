from conf import PROJECT_ROOT_DIR
import os
import pandas as pd
import numpy as np

from git_status import get_repo_list


def get_wiki_status_color(input_text):
    if input_text is None or input_text == 'inactive':
        result_text = ":heavy_multiplication_x:"
    else:
        result_text = ":heavy_check_mark:"
    return '<sub>{}</sub>'.format(result_text)


def get_wiki_rating(input_rating):
    result_text = ''
    if input_rating is not None and not np.isnan(input_rating):
        rating = int(input_rating)
        result_text = ':star:x{}'.format(rating)
    return '<sub>{}</sub>'.format(result_text)


def generate_wiki_per_category(output_path):
    """

    :param output_path:
    """
    repo_df = get_repo_list()
    for category in repo_df['category'].unique():
        category_df = repo_df[repo_df['category'] == category].copy()
        url_md_list = []
        for idx, irow in category_df[['name', 'url']].iterrows():
            url_md_list.append('<sub>[{}]({})</sub>'.format(irow['name'], irow['url']))

        formatted_df = pd.DataFrame({
            'repo': url_md_list,
            'comment': category_df['comment'].apply(lambda x: '<sub>{}</sub>'.format(x)),
            'created_at': category_df['created_at'].apply(lambda x: '<sub>{}</sub>'.format(x)),
            'last_commit': category_df['last_commit'].apply(lambda x: '<sub>{}</sub>'.format(x)),
            'star_count': category_df['star_count'].apply(lambda x: '<sub>{}</sub>'.format(x)),
            'repo_status': category_df['repo_status'],
            'rating': category_df['rating']
        })
        # add color for the status
        formatted_df['repo_status'] = formatted_df['repo_status'].apply(lambda x: get_wiki_status_color(x))
        formatted_df['rating'] = formatted_df['rating'].apply(lambda x: get_wiki_rating(x))

        output_path_full = os.path.join(output_path, '{}.md'.format(category))
        with open(output_path_full, 'w') as f:
            f.write(formatted_df.to_markdown(index=False))


if __name__ == '__main__':
    local_path = os.path.join(PROJECT_ROOT_DIR, 'generated_wiki')
    generate_wiki_per_category(local_path)
