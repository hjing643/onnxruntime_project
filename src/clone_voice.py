from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import torchaudio
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.api import TTS
import noisereduce as nr
import librosa
import soundfile as sf
from pathlib import Path
import traceback
from TTS.utils.manage import ModelManager



def denoise_audio(input_path, output_path):
    y, sr = librosa.load(input_path, sr=16000)
    noise_start=0.0
    noise_end=0.5
    noise_sample = y[int(noise_start * sr):int(noise_end * sr)]
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_noise, sr)

def xtts_clone():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    source_audio_path = "../filedepends/sounds/gene.wav"  # 一段 3-10 秒的干净人声 wav（16kHz）
    denoise_audio(source_audio_path, "../filedepends/sounds/output_denoised.wav")
    source_audio_path2 = "../filedepends/sounds/output_denoised.wav"
    try:
        tts.tts_to_file(
            text='''森林里住着一只聪明的小猴子。
            有一天，它看到一只老虎掉进了猎人的陷阱里，正在挣扎。老虎看见小猴子，就说：“小猴子，求求你救救我，我再也不吃小动物了。”
            小猴子有点犹豫，但还是把老虎救了出来。
            老虎一出来，就露出獠牙说：“我饿了，你正好是我的午餐！”
            小猴子赶紧说：“等等！我们去问问森林里最公平的大象吧。”
            大象听了之后说：“你们演一遍让我看看。”
            于是，老虎又跳进了陷阱，小猴子把陷阱盖上。
            大象笑着说：“你现在就留在里面吧，小猴子救了你一次，你却恩将仇报！”
            老虎无奈地呜咽着，而小猴子开心地跳上了树枝。''',
            speaker_wav=source_audio_path2,
            file_path="../filedepends/sounds/output_clone.wav",
            language="zh-cn",
            temperature=0.7
            )
    except Exception as e:  
        print("发生异常:", e)
        traceback.print_exc()
    print("finished")

if __name__ == "__main__":   
    xtts_clone()