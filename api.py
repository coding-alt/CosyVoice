import time
import torch
import logging
import asyncio
import uvicorn
import argparse
import torchaudio
from io import BytesIO
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cosyvoice.cli.cosyvoice import CosyVoice

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str = Query("欢迎使用TTS API", description="text to synthesize")
    spk_id: str = Query("中文女", description="speaker id")
    streaming: bool = Query(False, description="Whether to stream the audio output")

app = FastAPI()

cosyvoice = None
sft_spk = []
target_sr = 22050

async def process_segment(segment, spk_id):
    logger.debug(f"Processing segment: {segment}")
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(None, cosyvoice.inference_sft_not_split, segment, spk_id)
    return output['tts_speech']

@app.post("/")
async def generate_tts_audio(params: TTSRequest):
    if params.spk_id not in sft_spk:
        params.spk_id = '中文女'

    text_segments = cosyvoice.frontend.text_normalize(params.text, split=True)
    logger.debug(f"Text segments: {text_segments}")

    async def audio_generator():
        for segment in text_segments:
            start_time = time.time()
            tts_speech = await process_segment(segment, params.spk_id)
            end_time = time.time()
            generation_time = end_time - start_time
            logger.debug(f"TTS Speech Shape: {tts_speech.shape}, Generation Time: {generation_time:.4f} seconds")

            audio_io = BytesIO()
            torchaudio.save(audio_io, tts_speech, target_sr, format="wav")
            audio_io.seek(0)
            segment_audio = audio_io.read()
            yield segment_audio
            logger.debug(f"Segment audio sent: {len(segment_audio)} bytes")

            torch.cuda.empty_cache()

    if params.streaming:
        return StreamingResponse(audio_generator(), media_type="audio/wav")
    else:
        for i in range(0, len(text_segments), batch_size):
            batch = text_segments[i:i + batch_size]
            tasks = [process_segment(segment, params.spk_id) for segment in batch]
            all_tts_speeches = await asyncio.gather(*tasks)
            torch.cuda.empty_cache()
        
        combined_speech = torch.concat(all_tts_speeches, dim=1)
        logger.debug(f"TTS Speech Shape: {combined_speech.shape}")
        audio_io = BytesIO()
        torchaudio.save(audio_io, combined_speech, target_sr, format="wav")
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
    parser.add_argument("--batch_size", type=int, default=1, help='Number of segments to process in parallel')
    args = parser.parse_args()
    
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    batch_size = args.batch_size
    
    uvicorn.run(app, host=args.host, port=args.port)