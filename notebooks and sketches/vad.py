#!/usr/bin/env python3

import torch
# from IPython.display import Audio
# from pprint import pprint

print(torch.cuda.is_available())

torch.set_num_threads(1)
SAMPLING_RATE = 16000

# download example
# torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

print("reading wav")
wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)

# print("getting timestamps")
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
# pprint(speech_timestamps)

# print("saving audio")
# save_audio('only_speech.wav',
#            collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)

vad_iterator = VADIterator(model)
window_size_samples = 512 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    speech_dict = vad_iterator(wav[i: i+ window_size_samples], return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio