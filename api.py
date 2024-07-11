import uvicorn
import argparse
import torchaudio
from io import BytesIO
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import CosyVoice

class TTSRequest(BaseModel):
    text: str = Query("欢迎使用TTS API", description="text to synthesize")
    spk_id: str = Query("中文女", description="speaker id")

app = FastAPI()

cosyvoice = None
sft_spk = []
target_sr = 22050

@app.post("/")
async def generate_tts_audio(params: TTSRequest):
    if params.spk_id not in sft_spk:
        params.spk_id = '中文女'

    output = cosyvoice.inference_sft(params.text, params.spk_id)
    
    audio_io = BytesIO()
    torchaudio.save(audio_io, output['tts_speech'], target_sr, format="wav")
    audio_io.seek(0)

    return StreamingResponse(audio_io, media_type="audio/wav")

@app.get("/speaker")
async def get_speakers():
    return {'code': 200, 'msg': 'success', 'data': sft_spk}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model_dir", type=str, default='pretrained_models/CosyVoice-300M', help='local path or modelscope repo id')
    args = parser.parse_args()
    
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    
    uvicorn.run(app, host=args.host, port=args.port)