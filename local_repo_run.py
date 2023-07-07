from analyzer import Analyzer

if __name__ == "__main__":
    """
    Local run for Analyzer script
    """

    # Fetch files
    files = Analyzer.get_files_from_dir()

    # Iterate over files and process
    for _file, _content in Analyzer.read_files(files).items():
        code_analyzer = Analyzer( _content,_file)
        code_analyzer.analyze_file()

    print('All files processed.')
