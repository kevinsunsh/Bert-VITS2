"""Main entrypoint for the app."""

from io import BytesIO
import time
import json
import logging
from fastapi import Depends, FastAPI, Request, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    Response,
    HTMLResponse,
    JSONResponse,
)
from scipy.io import wavfile
import torch
from infer_utils import infer


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)


@app.post("/tts_gen")
async def genvideo(request: Request):
    inputs = await request.json()
    speaker = inputs["speaker"]
    text = inputs["text"].replace("/n", "")
    sdp_ratio = float(0.2)
    noise = float(0.5)
    noisew = float(0.6)
    length = float(1.2)
    language = inputs["language"]
    fmt = "wav"

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language,
            hps=hps,
            net_g=net_g,
            device=dev,
        )

    with BytesIO() as wav:
        wavfile.write(wav, hps.data.sampling_rate, audio)
        torch.cuda.empty_cache()
        return Response(wav.getvalue(), mimetype="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
