import base64
import glob
from datetime import datetime
from typing import Optional

import aiohttp
import numpy as np
import requests
from fastapi import WebSocket
from langchain import OpenAI, LLMChain, PromptTemplate
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from configs import Config


class Analyzer:
    main_prompt = """
    Firstly, give the following text an informative title. 
    Then, on a new line, write a 75-100 word summary of the following text:
    {text}
    Return your answer in the following format:
    Title | Summary...
    e.g. 
    Why Artificial Intelligence is Good | AI can make humans more productive by 
    automating many repetitive processes.
    TITLE AND CONCISE SUMMARY:
    """

    summarize_prompt = """Write a detailed summary on the structure of the provided 
       content which contains code from selected files from a Github repository, which 
       deploys a chatbot system in Microsoft Azure. Please list all necessary details 
       which can be extrapolated later to specific guidelines how to reverse engineer 
       the repository. I am specifically looking for answers on: 
       i. the specific steps to deploy this resource? Please list all the files that 
       contain the specific tasks that automate the deployment!
       ii. relevant files and code sections that I need to alter in case I want to 
       adjust the overall tool of the repository for my use case. Please list all 
       files and name the code sections.
       iii. the detailed steps that need to be performed in order to adjust this 
       repository as a project template for customized deployments.: 
       {text}
       """

    config = Config()

    def __init__(self, content: str, file: Optional[str] = None):
        self.map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        self.file = file
        self.content = content

    @staticmethod
    def get_files_from_dir() -> list:
        """
        Function to get files from local directory

        """
        files_list = []
        is_local = Analyzer.config.is_local
        file_patterns = Analyzer.config.file_patterns
        github_repo = Analyzer.config.github_repo
        dir_path = Analyzer.config.dir_path
        owner = Analyzer.config.owner
        repo_name = Analyzer.config.repo_name
        default_branch = Analyzer.config.default_branch

        if is_local:
            for pattern in file_patterns:
                files_list.extend(glob.glob(dir_path + '/' + pattern,
                                            recursive=True))
        else:
            all_files_endpoint = (
                f"https://api.github.com/repos/{owner}/"
                f"{repo_name}/git/trees/{default_branch}?recursive=1")

            response = requests.get(all_files_endpoint)
            if response.status_code == 200:
                repo = response.json()
                if repo.get('tree') is not None:
                    tree = repo.get('tree')
                    files_list = [
                        item['path'] for item in tree
                        if f"*.{item['path'].split('.')[-1]}" in file_patterns
                    ]
                else:
                    print("Repository tree is empty")
            else:
                print(f"Repository cannot bee accessed {github_repo}")
        return files_list

    @staticmethod
    async def aget_files_from_dir(github_repo: str, websocket: WebSocket) -> tuple:
        """
        Function to get files from local directory

        """
        files_list = []
        file_patterns = Analyzer.config.file_patterns
        repo_metadata = github_repo.split("github.com/")[-1]
        owner, repo_name = repo_metadata.split("/")
        default_branch = 'master'

        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(verify_ssl=False)) as aio_session:
            endpoint = f"https://api.github.com/repos/{owner}/{repo_name}"
            async with aio_session.get(endpoint) as response:
                if response.status == 200:
                    default_branch = (await response.json())['default_branch']

            all_files_endpoint = (
                f"https://api.github.com/repos/{owner}/"
                f"{repo_name}/git/trees/{default_branch}?recursive=1")

            async with aio_session.get(all_files_endpoint) as response:
                if response.status == 200:  # NOQA
                    repo = await response.json()
                    if repo.get('tree') is not None:
                        tree = repo.get('tree')
                        files_list = [
                            item['path'] for item in tree
                            if f"*.{item['path'].split('.')[-1]}" in file_patterns
                        ]
                    else:
                        message = "Repository tree is empty"
                        await websocket.send_text(message)
                        print(message)
                else:
                    message = f"Repository cannot bee accessed {github_repo}"
                    await websocket.send_text(message)
                    print(message)

        return files_list, owner, repo_name

    @staticmethod
    def read_files(file_paths: list) -> dict:
        """
        Function to read content of the files
        """
        contents_dict = {}
        is_local = Analyzer.config.is_local

        if is_local:
            for file_path in file_paths:
                with open(file_path, 'r') as f:
                    contents_dict[file_path] = f.read()
        else:
            owner = Analyzer.config.owner
            repo_name = Analyzer.config.repo_name
            for file_path in file_paths:
                file_content_api = (
                    f'https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}'
                )
                response = requests.get(file_content_api)
                if response.status_code == 200:
                    contents_dict[file_path] = (
                        base64.b64decode(response.json()['content']).decode(
                            'UTF-8')
                    )
        return contents_dict

    @staticmethod
    async def aread_files(file_paths: list, owner: str, repo_name: str) -> dict:
        """
        Function to read content of the files
        """
        contents_dict = {}
        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(verify_ssl=False)) as aio_session:
            for file_path in file_paths:
                file_content_api = (
                    f'https://api.github.com/repos/{owner}/{repo_name}/contents/{file_path}'
                )
                async with aio_session.get(file_content_api) as response:
                    if response.status == 200:
                        response = await response.json()
                        contents_dict[file_path] = (
                            base64.b64decode(response['content']).decode('UTF-8'))

        return contents_dict

    @staticmethod
    def get_chunks_from_text(text: str, num_chunks: int = 10) -> list:
        """
        Function to break a large text into chunks
        """

        words = text.split()
        words_per_chunk = len(words) // num_chunks
        chunks_list = []
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            chunks_list.append(chunk)
        return chunks_list

    @staticmethod
    async def aget_chunks_from_text(text: str, num_chunks: int = 10) -> list:
        """
        Function to break a large text into chunks
        """

        words = text.split()
        words_per_chunk = len(words) // num_chunks
        chunks_list = []
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            chunks_list.append(chunk)
        return chunks_list

    def summarize_chunks(self, chunks_list: list, template: PromptTemplate) -> list:
        """
        Function to summarize chunks_list using OpenAI
        """

        llm_chain = LLMChain(llm=self.map_llm, prompt=template)
        summaries = []
        for chunk in chunks_list:
            chunk_summary = llm_chain.apply([{'text': chunk}])
            summaries.append(f" {chunk_summary}")
        return summaries

    async def asummarize_chunks(self, chunks_list: list, template: str) -> list:
        """
        Function to summarize chunks_list using OpenAI
        """

        llm_chain = LLMChain(llm=self.map_llm, prompt=template)
        summaries = []
        for chunk in chunks_list:
            chunk_summary = llm_chain.apply([{'text': chunk}])
            summaries.append(f" {chunk_summary}")
        return summaries

    @staticmethod
    def create_similarity_matrix(chunks_list: list):
        """
        Function to calculate similarity matrix
        """

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(
            [' '.join(chunk.split()[:200]) for chunk in chunks_list])
        return cosine_similarity(vectors)

    @staticmethod
    async def acreate_similarity_matrix(chunks_list: list):
        """
        Function to calculate similarity matrix
        """

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(
            [' '.join(chunk.split()[:200]) for chunk in chunks_list])
        return cosine_similarity(vectors)

    @staticmethod
    def get_topics(similarity_matrix_, num_topics=5):
        """
        Get the topics from the similarity matrix
        """
        distances = 1 - similarity_matrix_
        kmeans = KMeans(n_clusters=num_topics).fit(distances)
        clusters = kmeans.labels_
        chunk_topics = [np.where(clusters == i)[0] for i in range(num_topics)]
        return chunk_topics

    @staticmethod
    async def aget_topics(similarity_matrix_, num_topics=5):
        """
        Get the topics from the similarity matrix
        """
        distances = 1 - similarity_matrix_
        kmeans = KMeans(n_clusters=num_topics).fit(distances)
        clusters = kmeans.labels_
        chunk_topics = [np.where(clusters == i)[0] for i in range(num_topics)]
        return chunk_topics

    @staticmethod
    def parse_title_summary_results(results: list) -> list:
        """
        Function to parse title and summary results
        """

        outputs = []
        for result in results:

            result = result.replace('\n', '')
            if '|' in result:
                processed = {'title': result.split('|')[0],
                             'summary': result.split('|')[1][1:]
                             }
            elif ':' in result:
                processed = {'title': result.split(':')[0],
                             'summary': result.split(':')[1][1:]
                             }
            elif '-' in result:
                processed = {'title': result.split('-')[0],
                             'summary': result.split('-')[1][1:]
                             }
            else:
                processed = {'title': '',
                             'summary': result
                             }
            outputs.append(processed)
        return outputs

    @staticmethod
    async def aparse_title_summary_results(results: list) -> list:
        """
        Function to parse title and summary results
        """

        outputs = []
        for result in results:

            result = result.replace('\n', '')
            if '|' in result:
                processed = {'title': result.split('|')[0],
                             'summary': result.split('|')[1][1:]
                             }
            elif ':' in result:
                processed = {'title': result.split(':')[0],
                             'summary': result.split(':')[1][1:]
                             }
            elif '-' in result:
                processed = {'title': result.split('-')[0],
                             'summary': result.split('-')[1][1:]
                             }
            else:
                processed = {'title': '',
                             'summary': result
                             }
            outputs.append(processed)
        return outputs

    def summarize_stage(self, chunks_list: list, topics_list: list) -> list:
        """
        Function to summarize the stage
        """

        print(f'Start time: {datetime.now()}')

        # Prompt to get title and summary for each topic

        map_prompt = PromptTemplate(template=Analyzer.summarize_prompt,
                                    input_variables=["text"])

        # Define the LLMs
        map_llm_chain = LLMChain(llm=self.map_llm, prompt=map_prompt)

        summaries = []
        for i in range(len(topics_list)):
            topic_summaries = []
            for topic in topics_list[i]:
                map_llm_chain_input = [{'text': chunks_list[topic]}]
                # Run the input through the LLM chain (works in parallel)
                map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)
                stage_1_outputs = Analyzer.parse_title_summary_results(
                    [e['text'] for e in map_llm_chain_results])
                # Split the titles and summaries
                topic_summaries.append(stage_1_outputs[0]['summary'])
            # Concatenate all summaries of a topic
            summaries.append(' '.join(topic_summaries))

        print(f'Stage done time {datetime.now()}')

        return summaries

    async def asummarize_stage(self, chunks_list: list, topics_list: list,
                               websocket: WebSocket) -> list:
        """
        Function to summarize the stage
        """
        await self.logger(f'Start time: {datetime.now()}', websocket)

        # Prompt to get title and summary for each topic

        map_prompt = PromptTemplate(template=Analyzer.summarize_prompt,
                                    input_variables=["text"])

        # Define the LLMs
        map_llm_chain = LLMChain(llm=self.map_llm, prompt=map_prompt)

        summaries = []
        for i in range(len(topics_list)):
            topic_summaries = []
            for topic in topics_list[i]:
                map_llm_chain_input = [{'text': chunks_list[topic]}]
                # Run the input through the LLM chain (works in parallel)
                map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)
                stage_1_outputs = await Analyzer.aparse_title_summary_results(
                    [e['text'] for e in map_llm_chain_results])
                # Split the titles and summaries
                topic_summaries.append(stage_1_outputs[0]['summary'])
            # Concatenate all summaries of a topic
            summaries.append(' '.join(topic_summaries))

        await self.logger(f'Stage done time {datetime.now()}', websocket)

        return summaries

    @staticmethod
    def get_prompt_template(template: str) -> PromptTemplate:
        return PromptTemplate(template=template,
                              input_variables=['text'])

    @staticmethod
    async def aget_prompt_template(template: str) -> PromptTemplate:
        return PromptTemplate(template=template,
                              input_variables=['text'])

    def analyze_file(self) -> None:
        print(f'Processing {self.file}...')

        print(f"Get chunks from {self.file}...")
        chunks = Analyzer.get_chunks_from_text(self.content)
        print("Chunks generated!")

        # Summarize chunks
        print("Summarizing chunks...")
        chunk_summaries = (
            self.summarize_chunks(
                chunks,
                self.get_prompt_template(Analyzer.main_prompt))
        )
        print("Chunks summarized!")

        # Create similarity matrix
        print("Creating similarity matrix...")
        similarity_matrix = Analyzer.create_similarity_matrix(chunks)
        print("Similarity matrix created!")

        # Get topics
        print("Getting topics...")
        topics = Analyzer.get_topics(similarity_matrix)
        print("Topics are got!")

        # Summarize stage
        print("Get stage summary...")
        stage_summary = self.summarize_stage(chunk_summaries, topics)

        print(f'Summary for {self.file}:\n{stage_summary}\n')

    async def aanalyze_file(self, websocket: WebSocket, is_github: bool = True) -> None:
        if is_github:
            await self.logger(f'Processing {self.file}...', websocket)
            await self.logger(f"Get chunks from {self.file}...", websocket)
        else:
            await self.logger(f'Processing...', websocket)
            await self.logger(f"Get chunks from input...", websocket)

        chunks = await Analyzer.aget_chunks_from_text(self.content)
        await self.logger("Chunks generated!", websocket)

        # Summarize chunks
        await self.logger("Summarizing chunks...", websocket)
        chunk_summaries = (await self.asummarize_chunks(
            chunks, await self.aget_prompt_template(Analyzer.main_prompt)))
        await self.logger("Chunks summarized!", websocket)

        # Create similarity matrix
        await self.logger("Creating similarity matrix...", websocket)
        similarity_matrix = await Analyzer.acreate_similarity_matrix(chunks)
        await self.logger("Similarity matrix created!", websocket)

        # Get topics
        await self.logger("Getting topics...", websocket)
        topics = await Analyzer.aget_topics(similarity_matrix)
        await self.logger("Topics are got!", websocket)

        # Summarize stage
        await self.logger("Get stage summary...", websocket)
        stage_summary = await self.asummarize_stage(chunk_summaries, topics, websocket)
        if is_github:
            await self.logger(f'Summary for {self.file}:\n{stage_summary}\n', websocket)
        else:
            await self.logger(f'Summary for input text:\n{stage_summary}\n', websocket)

    async def logger(self, message, websocket: WebSocket):
        print(message)
        await websocket.send_text(message)
