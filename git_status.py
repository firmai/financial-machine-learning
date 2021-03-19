import os
from conf import PROJECT_ROOT_DIR
import re

@DeprecationWarning
def parse_readme_md():
    file_path = os.path.join(PROJECT_ROOT_DIR, 'README.md')
    with open(file_path) as f:
        lines = f.readlines()[11:] # skip heading
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
                    print(link_line)
                    if len(link_line) > 0:
                        # usually in the format of '- [NAME](link) - comment
                        split_sections = link_line.split('- ')
                        if len(split_sections) == 2:
                            title_and_link = split_sections[1].strip()
                            title = re.search(r'\[(.*?)\]', title_and_link).group(1)
                            m_link = re.search(r'\((.*?)\)', title_and_link)
                            link_str = ''
                            if m_link is not None:
                                link_str = m_link.group(1)

                            pass
                        elif len(split_sections) == 3:
                            pass

                        print(split_sections)
                    line_num += 1
