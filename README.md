Code Analyzer 

Initial setup:
- Please pull the repository
- Create virtual environment by using following command: `python3.11 -m venv venv`, activate it `source venv/bin/activate` 
- Install dependencies with `pip install -r requirements.txt`
- Create the `.env` file

The `.env` file is actually a text file with a pairs `VARIABLE=VALUE`. This file should contain the following `VARIABLE=VALUE` pairs:
- OPENAI_API_KEY=<open_ai_key_placed_here>
- LOCAL_REPO_PATH=<path_to_your_local_repository>
- LOCAL_REPO=True/False
- GITHUB_REPO=<github_url>