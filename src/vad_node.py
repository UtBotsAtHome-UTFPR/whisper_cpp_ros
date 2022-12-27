#!/usr/bin/env python3

# ROS
import rospy
import rospkg
from audio_common_msgs.msg import AudioData, AudioInfo

# Sound processing
import wave
from scipy.io import wavfile
import noisereduce

# Other
from io import BytesIO
import numpy as np

# Torch
import torch
torch.set_num_threads(1)

'''
    TODO
        - store last 5 audios in logs
        - make param_speech_offset work before voice was detected
        - use clips from when no voice is detected as noise suppression samples
'''

class VadNode:

    # Init
    def __init__(self):
        rospy.init_node('vad_node', anonymous=True)
        rospy.loginfo("[vad] Node init")

        # Parameters
        self.param_acceptable_vad_confidence    = 0.8   # Between 0 and 1
        self.param_speech_offset                = 2     # Time to close speech after it is detected as False
        self.param_use_noise_suppresion               = False # 

        self.param_acceptable_vad_confidence = rospy.get_param(
            "~acceptable_vad_confidence", 0.8)
        self.param_speech_offset = rospy.get_param(
            "~speech_offset", 2)
        self.param_use_noise_suppresion = rospy.get_param(
            "~use_noise_suppresion", False)
        rospy.loginfo("[vad] acceptable_vad_confidence : {}".format(self.param_acceptable_vad_confidence))
        rospy.loginfo("[vad] param_speech_offset : {}".format(self.param_speech_offset))
        rospy.loginfo("[vad] param_use_noise_suppresion : {}".format(self.param_use_noise_suppresion))


        # VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir     = 'snakers4/silero-vad',
            model           = 'silero_vad',
            force_reload    = False,
            onnx            = False)
        (self.get_speech_timestamps,
        self.save_audio,
        self.read_audio,
        self.VADIterator,
        self.collect_chunks) = utils

        # Gets path of this package
        self.packagePath = rospkg.RosPack().get_path('utbots_voice')
        rospy.loginfo("[vad] Package path: {}".format(self.packagePath))

        # To store audio frames
        self.audio_msg_chunks   = []
        self.speech_chunks      = []
        self.t_last             = rospy.get_time()
        self.t_interval         = 0.250
        rospy.loginfo("[vad] Interval between sample evaluation: {}s".format(self.t_interval))

        # Defines if speech is happening
        self.is_speech_happening    = False
        self.last_t_speech          = rospy.get_time()

        # Defines if it is time to close speech chunk
        self.speech_open    = False
        self.speech_close   = False

        # Topics
        top_audio_info      = "/audio_info"
        top_audio_input     = "/audio"
        top_audio_output    = "/audio/voice"

        # Publishers
        self.pub_audio_output = rospy.Publisher(top_audio_output, AudioData, queue_size=1)

        # Subscribers
        self.sub_audio_raw = rospy.Subscriber(
            top_audio_input, AudioData, self.Callback_audio, queue_size=1000)

        # Waits for audio info message before doing anything else
        rospy.loginfo("[vad] Waiting for audio info")
        self.SetAudioInfo(rospy.wait_for_message(top_audio_info, AudioInfo))

        # Loop
        self.loopRate = rospy.Rate(30)
        self.MainLoop()

    ''' Writes audio chunks to a wave file '''
    def WriteAudioWavesToFile(self, filePath, chunks):
        with wave.open(filePath, 'wb') as file:
            width       = 2
            comptype    = "NONE"
            compname    = "noncompressed"
            file.setparams((self.channels, width, self.sample_rate, len(chunks), comptype, compname))
            byte_count = 0
            for chunk in chunks:
                byte_count = byte_count + len(chunk)
                file.writeframes(chunk)
            # rospy.loginfo("[vad] Copied {} data bytes to file".format(byte_count))

    ''' Returns frames from wave file '''
    def ReadAudioWavesFromFile(self, filepath):
        with wave.open(filepath, 'rb') as file:
            return file.readframes(file.getnframes())

    ''' Returns number of frames in a wave file '''
    def GetNFrames(self, filepath):
        with wave.open(filepath, 'rb') as file:
            return file.getnframes()

    ''' Returns time in seconds '''
    def GetSecondsFromNFrames(self, nframes, framerate):
        return nframes / float(framerate)

    ''' Returns list with concatenated chunks '''
    def ConcatenateAudioChunks(self, chunks):
        # rospy.loginfo("[vad] Concatenating audio chunks")
        tmp_file = BytesIO()
        self.WriteAudioWavesToFile(tmp_file, chunks)
        tmp_file.seek(0)
        return self.ReadAudioWavesFromFile(tmp_file)
    
    ''' Converts np.int16 sound to float32 '''
    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound   = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/abs_max
        sound   = sound.squeeze()
        return sound
    
    ''' Determines whether or not speech is happening '''
    def EvaluateSpeechPresence(self, chunks):

        tmp_wav_file = BytesIO()
        self.WriteAudioWavesToFile(tmp_wav_file, chunks)
        tmp_wav_file.seek(0)

        with wave.open(tmp_wav_file, 'rb') as file:
            chunk_bytes = file.readframes(file.getnframes())
            try:
                chunk_int16             = np.frombuffer(chunk_bytes, np.int16)
                chunk_float32           = self.int2float(chunk_int16)
                voice_confidence        = self.model(torch.from_numpy(chunk_float32), 16000).item()
                backup_speech_status    = self.is_speech_happening

                if voice_confidence > self.param_acceptable_vad_confidence:
                    self.speech_open            = True
                    self.is_speech_happening    = True
                    self.last_t_speech          = rospy.get_time()
                else:
                    self.is_speech_happening = False
                    if rospy.get_time() - self.last_t_speech > self.param_speech_offset and self.speech_open == True:
                        self.speech_close = True

                if backup_speech_status != self.is_speech_happening:
                    rospy.loginfo("[vad] Speech happening: {}".format(self.is_speech_happening))
                        
            except:
                rospy.loginfo("[vad] Exception occured in VAD evaluation")

    ''' Applies noise suppresion to a wav clip'''
    def SuppressNoise(self, chunks):
        tmp_file = BytesIO()
        self.WriteAudioWavesToFile(tmp_file, chunks)
        tmp_file.seek(0)
        sampleRate, sampleData = wavfile.read(tmp_file)
        wav_denoised = noisereduce.reduce_noise(y=sampleData, sr=sampleRate, stationary=False)
        tmp_file_denoised = BytesIO()
        wavfile.write(tmp_file_denoised, sampleRate, wav_denoised)
        tmp_file_denoised.seek(0)
        return self.ReadAudioWavesFromFile(tmp_file_denoised)

    ''' Publishes chunks through ROS topic '''
    def PublishChunks(self, chunks):
        msg_audio = AudioData()
        tmp_file = BytesIO()
        self.WriteAudioWavesToFile(tmp_file, chunks)
        tmp_file.seek(0)
        msg_audio.data = tmp_file.read()
        self.pub_audio_output.publish(msg_audio)

    ''' Determines whether or not an ongoing speech is finished '''
    def EvaluateSpeechClosure(self, chunks):
        if self.speech_close == True:
            rospy.loginfo("[vad] Closing speech")
            self.speech_open    = False
            self.speech_close   = False
            if self.param_use_noise_suppresion == True:
                self.speech_chunks = self.SuppressNoise(self.speech_chunks)
            self.PublishChunks(self.speech_chunks)
            self.speech_chunks.clear()
        else:
            if self.speech_open == True:
                self.speech_chunks = self.speech_chunks + chunks
   
    ''' Callback function for audio messages '''
    def Callback_audio(self, msg):
        t_now   = rospy.get_time()
        dt      = t_now - self.t_last

        if (dt > self.t_interval):
            self.t_last = t_now

            # Puts together the audio chunks
            chunks = [self.audio_msg_chunks[i].data for i in range(0, len(self.audio_msg_chunks))]
            
            tmp_file = BytesIO()
            self.WriteAudioWavesToFile(tmp_file, chunks)
            tmp_file.seek(0)
            chunks_duration = self.GetSecondsFromNFrames(self.GetNFrames(tmp_file), self.sample_rate)

            # Only process if already has certain amount of chunks
            if chunks_duration > 0.2:
                # rospy.loginfo("[vad] Time since last chunk evaluation: {0:.3f}s".format(dt))

                concatenatedChunks = [self.ConcatenateAudioChunks(chunks)]

                # Determines if speech is happening
                self.EvaluateSpeechPresence(concatenatedChunks)

                # Closes or keeps stacking chunks to current speech
                self.EvaluateSpeechClosure(chunks)

                # rospy.loginfo("[vad] Clearing audio_msg_chunks buffer")
                self.audio_msg_chunks.clear()

                # rospy.loginfo("[vad] Took {0:.3f}s".format(rospy.get_time() - t_now))
        else:
            self.audio_msg_chunks.append(msg)

    def SetAudioInfo(self, msg_info):
        self.channels = msg_info.channels
        self.sample_rate = msg_info.sample_rate
        self.sample_format = msg_info.sample_format
        self.bitrate = msg_info.bitrate
        self.coding_format = msg_info.coding_format
        rospy.loginfo("[vad] Channels: {}".format(self.channels))
        rospy.loginfo("[vad] Sample rate: {}".format(self.sample_rate))
        rospy.loginfo("[vad] Sample format: {}".format(self.sample_format))
        rospy.loginfo("[vad] Bitrate: {}".format(self.bitrate))
        rospy.loginfo("[vad] Coding format: {}".format(self.coding_format))

    # Main loop
    def MainLoop(self):
        print("-")
        rospy.loginfo("[vad] Looping")
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

if __name__ == "__main__":
    VadNode()
