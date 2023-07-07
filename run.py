import os
from pathlib import Path

import openai
import uvicorn
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from analyzer import Analyzer

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
log_file = "Your Results"

port = os.environ.get("PORT") or 8000
host_url = os.environ.get("HOST") or "code-analyzer.azurewebsites.net"

app = FastAPI(title='Text Analyzer Demo')


@app.get("/")
async def get(request: Request):
    """Log file viewer

    Args:
        request (Request): Default web request.

    Returns:
        TemplateResponse: Jinja template with context data.
    """
    context = {"title": "Text Analyzer Demo",
               "log_file": log_file,
               "host_url": host_url}
    return templates.TemplateResponse("index.html",
                                      {"request": request, "context": context})


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for any message from the client
            params = await websocket.receive_json()
            input = params.get('github_repo')
            openai_key = params.get('openai_key')
            is_http = input.strip().lower().startswith('http')
            is_url = True if input is not None and is_http else False
            is_openai_key = True if openai_key is not None and '-' in openai_key else False
            if is_url and is_openai_key:
                await websocket.send_text(
                    "Details saved. Please do not reload the page!"
                )
                await websocket.send_text("Set OpenAI secret key!")
                openai.api_key = openai_key
                os.environ['OPENAI_API_KEY'] = openai_key

                await websocket.send_text("Analyzing...")
                await websocket.send_text("Fetching files from public repository...")

                files, owner, repo_name = await Analyzer.aget_files_from_dir(
                    input, websocket
                )
                await websocket.send_text("Files are fetched!")
                await websocket.send_text("Iterating over fetched files...")
                if files and owner and repo_name:
                    files = await Analyzer.aread_files(file_paths=files,
                                                       owner=owner,
                                                       repo_name=repo_name)
                    items = files.items()
                    for _file, _content in items:
                        code_analyzer = Analyzer(_content, _file)
                        await code_analyzer.aanalyze_file(websocket)

                    await websocket.send_text("All files processed!")
                else:
                    await websocket.send_text("Something happen with the API")
            elif not is_url and is_openai_key:
                code_analyzer = Analyzer(input)
                await code_analyzer.aanalyze_file(websocket, is_github=False)
            else:
                await websocket.send_text("Please check provided details")

            await websocket.send_text('Analyzing completed.')

    except WebSocketDisconnect as e:
        print('error:', e)
        await websocket.send_text(f'error:{e}')


if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True,
        workers=1,
    )
