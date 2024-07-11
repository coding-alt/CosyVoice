import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append('{}/third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from hyperpyyaml import load_hyperpyyaml
import torch, torchaudio
import fnmatch, argparse, librosa

max_val = 0.8
prompt_sr, target_sr = 16000, 22050
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
from typing import Callable

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

class Trainer(CosyVoiceFrontEnd):

    def __init__(self, model_dir):
        instruct = False
        print('{}/cosyvoice.yaml'.format(model_dir))
        self.model_dir = model_dir
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        super().__init__(configs['get_tokenizer'],
                         configs['feat_extractor'],
                         '{}/campplus.onnx'.format(model_dir),
                         '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                         '{}/spk2info.pt'.format(model_dir),
                         instruct,
                         configs['allowed_special'])

    def start(self, dest):
        pattern = "*.wav"
        for root, dirs, files in os.walk(dest):
            for filename in fnmatch.filter(files, pattern):
                a = filename.replace(".wav", "").split("#")
                spk_name = os.path.basename(root)
                if self.spk2info.get(spk_name):
                    break
                print(f"开始训练:{spk_name}")
                prompt_speech_16k = postprocess(load_wav(os.path.join(root, filename), prompt_sr))
                prompt_text_token, prompt_text_token_len = self._extract_text_token(a[-1])
                prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
                speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_22050)
                speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)

                embedding = self._extract_spk_embedding(prompt_speech_16k)

                self.spk2info[spk_name] = {
                    "embedding": embedding,
                    "speech_feat": speech_feat,
                    "speech_token": speech_token
                }
                print(f"训练完成:{spk_name}")

    def save(self):
        torch.save(self.spk2info, os.path.join(self.model_dir, "spk2info.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice-300M',
                        help='local path or modelscope repo id')
    parser.add_argument('--audio_dir',
                        type=str,
                        default='',
                        help='输入一个目录,目录名对应模型名。')
    args = parser.parse_args()
    trainer = Trainer(args.model_dir)

    trainer.start(args.audio_dir)
    trainer.save()