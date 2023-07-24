import asyncio
import base64
import glob
import re
from datetime import datetime
from typing import Optional

import aiohttp
import networkx as nx
import numpy as np
import requests
from fastapi import WebSocket
from langchain import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from networkx.algorithms import community
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from configs import Config


class Analyzer:
    main_prompt = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
  and are different from each other:
  {text}

  Return your answer in a numbered list, with new line separating each title: 
  1. Title 1
  2. Title 2
  3. Title 3

  TITLES:
  """

    summarize_prompt = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text:
  {text}

  Return your answer in the following format:
  Title | Summary...
  e.g. 
  Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

  TITLE AND CONCISE SUMMARY:"""

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
                        await asyncio.sleep(0.01)
                        print(message)
                else:
                    message = f"Repository cannot bee accessed {github_repo}"
                    await websocket.send_text(message)
                    await asyncio.sleep(0.01)
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
    async def acreate_sentences(segments, MIN_WORDS, MAX_WORDS):
        # Combine the non-sentences together
        sentences = []

        is_new_sentence = True
        sentence_length = 0
        sentence_num = 0
        sentence_segments = []

        for i in range(len(segments)):
            if is_new_sentence == True:
                is_new_sentence = False
            # Append the segment
            sentence_segments.append(segments[i])
            segment_words = segments[i].split(' ')
            sentence_length += len(segment_words)

            # If exceed MAX_WORDS, then stop at the end of the segment
            # Only consider it a sentence if the length is at least MIN_WORDS
            if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or \
                    sentence_length >= MAX_WORDS:
                sentence = ' '.join(sentence_segments)
                sentences.append({
                    'sentence_num': sentence_num,
                    'text': sentence,
                    'sentence_length': sentence_length
                })
                # Reset
                is_new_sentence = True
                sentence_length = 0
                sentence_segments = []
                sentence_num += 1

        return sentences

    @staticmethod
    async def aget_chunks_from_text(text: str, num_chunks: int = 10) -> list:
        """
        Function to break a large text into chunks
        """

        text = text.replace("\n", " ")
        segments = text.split('.')
        segments = [segment + '.' for segment in segments]
        # Further split by comma
        segments = [segment.split(',') for segment in segments]
        # Flatten
        segments = [item for sublist in segments for item in sublist]
        sentences = await Analyzer.acreate_sentences(segments, MIN_WORDS=20,
                                                     MAX_WORDS=80)
        CHUNK_LENGTH = 5
        STRIDE = 1
        chunks_list = []
        for i in range(0, len(sentences), (CHUNK_LENGTH - STRIDE)):
            chunk = ' '.join([item['text'] for item in sentences[i:i + CHUNK_LENGTH]])
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
    async def acreate_similarity_matrix(num_1_chunks: int, summary_embeds: dict):
        """
        Function to calculate similarity matrix
        """

        # Get similarity matrix between the embeddings of the chunk summaries
        summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
        summary_similarity_matrix[:] = np.nan

        for row in range(num_1_chunks):
            for col in range(row, num_1_chunks):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])
                summary_similarity_matrix[row, col] = similarity
                summary_similarity_matrix[col, row] = similarity
        return summary_similarity_matrix

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
    async def aget_topics(title_similarity, num_topics=8, bonus_constant=0.25,
                          min_size=3):

        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = 1 / (
                        abs(row - col)) * bonus_constant

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        desired_num_topics = num_topics
        # Store the accepted partitionings
        topics_title_accepted = []

        resolution = 0.85
        resolution_step = 0.01
        iterations = 40

        # Find the resolution that gives the desired number of topics
        topics_title = []
        while len(topics_title) not in [desired_num_topics, desired_num_topics + 1,
                                        desired_num_topics + 2]:
            topics_title = community.louvain_communities(title_nx_graph,
                                                         weight='weight',
                                                         resolution=resolution)
            resolution += resolution_step

        lowest_sd_iteration = 0
        # Set lowest sd to inf
        lowest_sd = float('inf')

        for i in range(iterations):
            topics_title = community.louvain_communities(title_nx_graph,
                                                         weight='weight',
                                                         resolution=resolution)

            # Check SD
            topic_sizes = [len(c) for c in topics_title]
            sizes_sd = np.std(topic_sizes)

            topics_title_accepted.append(topics_title)

            if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        # Set the chosen partitioning to be the one with highest modularity
        topics_title = topics_title_accepted[lowest_sd_iteration]
        print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

        topic_id_means = [sum(e) / len(e) for e in topics_title]
        # Arrange title_topics in order of topic_id_means
        topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title),
                                                   key=lambda pair: pair[0])]
        # Create an array denoting which topic each chunk belongs to
        chunk_topics = [None] * title_similarity.shape[0]
        for i, c in enumerate(topics_title):
            for j in c:
                chunk_topics[j] = i

        return {
            'chunk_topics': chunk_topics,
            'topics': topics_title
        }

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

    async def asummarize_stage(self, chunks_list: list,
                               websocket: WebSocket) -> dict:
        """
        Function to summarize the stage
        """
        await self.logger(f'Start time: {datetime.now()}', websocket)

        # Prompt to get title and summary for each topic

        map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text:
          {text}

          Return your answer in the following format:
          Title | Summary...
          e.g. 
          Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

          TITLE AND CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template,
                                    input_variables=["text"])

        # Define the LLMs
        map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        map_llm_chain = LLMChain(llm=map_llm, prompt=map_prompt)
        map_llm_chain_input = [{'text': t} for t in chunks_list]
        # Run the input through the LLM chain (works in parallel)
        map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

        stage_1_outputs = await Analyzer.aparse_title_summary_results(
            [e['text'] for e in map_llm_chain_results])

        print(f'Stage 1 done time {datetime.now()}')

        return {
            'stage_1_outputs': stage_1_outputs
        }

    @staticmethod
    def get_prompt_template(template: str) -> PromptTemplate:
        return PromptTemplate(template=template,
                              input_variables=['text'])

    @staticmethod
    async def aget_prompt_template(template: str) -> PromptTemplate:
        return PromptTemplate(template=template,
                              input_variables=['text'])

    @staticmethod
    async def asummarize_stage_2(stage_1_outputs, topics, summary_num_words=250):
        print(f'Stage 2 start time {datetime.now()}')

        map_prompt_template = """Write a 75-100 word summary of the following text:
        {text}

        CONCISE SUMMARY:"""

        combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information. Finish your answer:
      {text}
      """ + str(summary_num_words) + """-WORD SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template,
                                    input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template,
                                        input_variables=["text"])

        topics_data = []
        for c in topics:
            topic_data = {
                'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
            }
            topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
            topics_data.append(topic_data)

        # Get a list of each community's summaries (concatenated)
        topics_summary_concat = [c['summaries_concat'] for c in topics_data]

        map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        reduce_llm = OpenAI(temperature=0, model_name='text-davinci-003', max_tokens=-1)

        # Run the map-reduce chain
        docs = [Document(page_content=t) for t in topics_summary_concat]
        chain = load_summarize_chain(chain_type="map_reduce", map_prompt=map_prompt,
                                     combine_prompt=combine_prompt,
                                     return_intermediate_steps=True,
                                     llm=map_llm, reduce_llm=reduce_llm)

        output = chain({"input_documents": docs}, return_only_outputs=True)

        final_summary = output['output_text']

        out = {
            'final_summary': final_summary
        }
        print(f'Stage 2 done time {datetime.now()}')

        return out

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

    async def aanalyze_file(self, websocket: WebSocket,
                            is_github: bool = True) -> Optional[str]:
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
        chunk_summaries = await self.asummarize_stage(chunks, websocket)
        await self.logger("Chunks summarized!", websocket)

        stage_1_outputs = chunk_summaries['stage_1_outputs']
        # Split the titles and summaries
        stage_1_summaries = [e['summary'] for e in stage_1_outputs]
        num_1_chunks = len(stage_1_summaries)
        # Create similarity matrix
        openai_embed = OpenAIEmbeddings()

        summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
        await self.logger("Creating similarity matrix...", websocket)
        similarity_matrix = await Analyzer.acreate_similarity_matrix(num_1_chunks,
                                                                     summary_embeds)
        await self.logger("Similarity matrix created!", websocket)

        # Get topics
        await self.logger("Getting topics...", websocket)
        num_topics = min(int(num_1_chunks / 4), 8)
        topics_out = await Analyzer.aget_topics(similarity_matrix,
                                                num_topics=num_topics,
                                                bonus_constant=0.2)
        topics = topics_out['topics']

        await self.logger("Topics are got!", websocket)

        # Summarize stage
        await self.logger("Get stage summary...", websocket)

        out = await Analyzer.asummarize_stage_2(stage_1_outputs, topics,
                                                summary_num_words=250)
        final_summary = out['final_summary']
        if is_github:
            await self.logger(f'Summary for {self.file}:\n{final_summary}\n', websocket)
            return final_summary
        else:
            await self.logger(f'Summary for input text:\n{final_summary}\n', websocket)

    async def logger(self, message, websocket: WebSocket):
        print(message)
        await websocket.send_text(message)
        await asyncio.sleep(0.01)
