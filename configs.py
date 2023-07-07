import os

import openai
import requests
from dotenv import dotenv_values


class Config:
    def __init__(self):
        config = dotenv_values('.env')
        openai.api_key = config.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = config.get('OPENAI_API_KEY', "")

        # Paths and patterns
        self.dir_path = config.get('LOCAL_REPO_PATH')
        self.file_patterns = ['*.json', '*.txt', '*.py', '*.md']
        self.is_local = True
        if config.get('LOCAL_REPO') == 'False':
            self.is_local = False
        self.github_repo = config.get('GITHUB_REPO')

        if self.github_repo is not None:
            repo_metadata = self.github_repo.split("github.com/")[-1]
            self.owner, self.repo_name = repo_metadata.split("/")
            response = requests.get(f"https://api.github.com/repos/"
                                    f"{self.owner}/{self.repo_name}")

            self.default_branch = 'master'
            if response.status_code == 200:
                self.default_branch = response.json()['default_branch']
