import os
import sys
import tempfile
import gradio as gr
import torch
import torchaudio
import librosa
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

max_val = 0.8
prompt_sr, target_sr = 16000, 22050

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech.squeeze().numpy(), top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    speech = torch.tensor(speech).unsqueeze(0)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.cat([speech, torch.zeros((1, int(target_sr * 0.2)))], dim=1)
    return speech

class Trainer(CosyVoiceFrontEnd):

    def __init__(self, model_dir):
        instruct = False
        print(f'{model_dir}/cosyvoice.yaml')
        self.model_dir = model_dir
        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f)
        super().__init__(configs['get_tokenizer'],
                         configs['feat_extractor'],
                         f'{model_dir}/campplus.onnx',
                         f'{model_dir}/speech_tokenizer_v1.onnx',
                         f'{model_dir}/spk2info.pt',
                         instruct,
                         configs['allowed_special'])

    def start(self, audio_files, state):
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            spk_name = os.path.splitext(filename)[0]
            if self.spk2info.get(spk_name):
                state.append(f"{spk_name} 已存在，跳过。")
                continue

            state.append(f"开始克隆:{spk_name}")

            try:
                speech, sr = torchaudio.load(audio_file)
                if sr != prompt_sr:
                    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=prompt_sr)(speech)

                if speech.shape[1] > 15 * prompt_sr:
                    speech = speech[:, :15 * prompt_sr]

                prompt_speech_16k = postprocess(speech)
                prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
                speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
                speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)

                embedding = self._extract_spk_embedding(prompt_speech_16k)

                self.spk2info[spk_name] = {
                    "embedding": embedding,
                    "speech_feat": speech_feat,
                    "speech_token": speech_token
                }
                state.append(f"克隆完成:{spk_name}")
            except Exception as e:
                error_msg = f"处理文件 {filename} 时出错: {str(e)}"
                state.append(error_msg)
        
        temp_model_path = self.save()
        return state, temp_model_path

    def save(self):
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, "spk2info.pt")
        torch.save(self.spk2info, temp_model_path)
        return temp_model_path

def train_model(audio_files, state):
    state.append(f"开始克隆音色...")
    trainer = Trainer(model_dir)
    state, model_path = trainer.start(audio_files, state)
    return state, model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    model_dir = args.model_dir

    css = """footer {visibility: hidden}"""
    with gr.Blocks(title="CosyVoice 音色克隆", css=css, theme="Kasien/ali_theme_custom") as demo:
        gr.Markdown("# CosyVoice 音色克隆")
        gr.Markdown("上传多个音频文件克隆音色。文件名将作为音色名，超过15秒的音频将自动裁剪。")
        
        with gr.Row():
            audio_files = gr.Files(label="上传多个WAV音频文件 (5-10秒)，注意采样率不低于16khz", file_count="multiple", file_types=["wav"])
            output = gr.Textbox(label="运行日志", lines=20)
            model_output = gr.File(label="输出合并后的音色模型")
        
        state = gr.State([])

        def update_output(state):
            return "\n".join(state)
        
        train_button = gr.Button("开始克隆")
        train_button.click(fn=train_model, inputs=[audio_files, state], outputs=[state, model_output], show_progress=True).then(
            fn=update_output, inputs=state, outputs=output
        )

    demo.launch(server_name="0.0.0.0")