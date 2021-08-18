import audioop
import wave
from timeit import default_timer as timer

import numpy as np
from deepspeech import Model
from jumpcutter.clip import Clip
from moviepy.editor import *
from pydub import AudioSegment

LANGUAGE = 'DE'

MODEL_PATH = ('deepspeech-0.9.3-models.pbmm', 'output_graph.pbmm')[LANGUAGE == 'DE']
SCORER_PATH = ('deepspeech-0.9.3-models.scorer', 'kenlm.scorer')[LANGUAGE == 'DE']

WPM = 200


def get_wav_from_video(filename):
    audioclip = AudioFileClip('res/' + filename + '.mp4')
    audioclip.write_audiofile('res/' + filename + '_audio.wav')

    audio_file = wave.open('res/' + filename + '_audio.wav', 'r')
    audio_data = audio_file.readframes(audio_file.getnframes())
    og_frames = audio_file.getnframes()
    og_rate = audio_file.getframerate()
    og_duration = og_frames / float(og_rate)

    out_file = wave.open('res/' + filename + '_audio_16k_stereo.wav', 'w')
    out_file.setnchannels(1)
    out_file.setparams((2, 2, 16000, 0, 'NONE', 'Uncompressed'))
    converted_audio = audioop.ratecv(audio_data, 2, 2, og_rate, 16000, None)
    out_file.writeframes(converted_audio[0])
    out_file.close()
    audio_file.close()

    sound = AudioSegment.from_wav('res/' + filename + '_audio_16k_stereo.wav')
    sound = sound.set_channels(1)
    sound.export('res/' + filename + '_audio_16k.wav', format="wav")

    return 'res/' + filename + '_audio_16k.wav', og_duration


def get_string_from_wav(wav_path):
    model_load_start = timer()
    ds = Model(MODEL_PATH)
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    scorer_load_start = timer()
    ds.enableExternalScorer(SCORER_PATH)
    scorer_load_end = timer() - scorer_load_start
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end), file=sys.stderr)

    fin = wave.open(wav_path, 'rb')
    fs_orig = fin.getframerate()

    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1 / fs_orig)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    string = ds.stt(audio)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

    return string


def cut_out_silent_parts(path):
    input_path = 'res/' + path + '.mp4'
    output_path = 'res/' + path + '_voiced.mp4'

    cuts = ["silent"]
    codec = None
    bitrate = None

    clip = Clip(str(input_path), -1, None)
    outputs = clip.jumpcut(
        cuts,
        0.02,
        0.5,
        0.1,
        0.1,
    )
    for cut_type, jumpcutted_clip in outputs.items():
        if len(outputs) == 2:
            jumpcutted_clip.write_videofile(
                str(
                    output_path.parent
                    / f"{output_path.stem}_{cut_type}_parts_cutted{output_path.suffix}"
                ),
                codec=codec,
                bitrate=bitrate,
            )
        else:
            jumpcutted_clip.write_videofile(
                str(output_path), codec=codec, bitrate=bitrate
            )


def speed_up_video(input_file, string, duration):
    word_count = len(string.split(' '))
    print("Word count: %i" % word_count)

    og_wpm = 60.0 * word_count / duration

    print("Original Words Per Minute after cutting out silent Parts: %.2f" % og_wpm)

    speed_factor = WPM / og_wpm

    input_path = 'res/' + input_file + '.mp4'
    output_path = 'res/' + input_file + '_custom_speed.mp4'

    os.system('ffmpeg -i {} -filter_complex "[0:v]setpts={}*PTS[v];[0:a]atempo={}[a]" -map "[v]" -map "[a]" {}'.format(
        input_path, (1 / speed_factor), speed_factor, output_path))


def remove_video_files(name):
    os.remove('res/' + name + '_voiced.mp4')
    os.remove('res/' + name + '_voiced_audio.wav')
    os.remove('res/' + name + '_voiced_audio_16k.wav')
    os.remove('res/' + name + '_voiced_audio_16k_stereo.wav')


def process_single_video(name):
    cut_out_silent_parts(name)

    wav_out_path, video_duration = get_wav_from_video(name + '_voiced')
    print('Duration: %.2fs' % video_duration)
    out_string = get_string_from_wav(wav_out_path)
    print(out_string)
    print(len(out_string.split(' ')))

    speed_up_video(name + '_voiced', out_string, video_duration)

    remove_video_files(name)


def get_file_name(name):
    return name.split('.')[0]


def process_videos_in_dir(dir_name):
    files = list(map(get_file_name, os.listdir('res/' + dir_name)))
    for f in files:
        print('Working on file: ' + f)
        process_single_video(dir_name + '/' + f)


process_videos_in_dir('convert')
