#!/usr/bin/env python3
import rospy
import rospkg
from audio_common_msgs.msg import AudioData, AudioInfo
import numpy as np
import wave

class TesterNode:

    # Init
    def __init__(self):
        rospy.init_node('audio_tester', anonymous=True)
        rospy.loginfo("[audio_tester] Node init")

        # Gets path of this package
        self.packagePath = rospkg.RosPack().get_path('utbots_voice')
        print("-")
        rospy.loginfo("[audio_tester] Package path: {}".format(self.packagePath))

        # To store audio frames
        self.audio_array = []
        self.t_last = rospy.get_time()
        self.t_interval = 10
        print("-")
        rospy.loginfo("[audio_tester] Interval between sample evaluation: {}s".format(self.t_interval))

        # Topics
        top_audio_info = "/audio_info"
        top_audio_raw = "/audio"
        top_audio_preprocessed = "/audio/voice"

        # Publishers
        self.pub_audio_preprocessed = rospy.Publisher(top_audio_preprocessed, AudioData, queue_size=1)

        # Subscribers
        self.sub_audio_raw = rospy.Subscriber(
            top_audio_raw, AudioData, self.Callback_audio, queue_size=1000)

        # Waits for audio info message before doing anything else
        print("-")
        rospy.loginfo("[audio_tester] Waiting for audio info")
        self.SetAudioInfo(rospy.wait_for_message(top_audio_info, AudioInfo))
        rospy.loginfo("[audio_tester] Channels: {}".format(self.channels))
        rospy.loginfo("[audio_tester] Sample rate: {}".format(self.sample_rate))
        rospy.loginfo("[audio_tester] Sample format: {}".format(self.sample_format))
        rospy.loginfo("[audio_tester] Bitrate: {}".format(self.bitrate))
        rospy.loginfo("[audio_tester] Coding format: {}".format(self.coding_format))
        rospy.loginfo("[audio_tester] File path: {}".format(self.filePath))

        # Loop
        self.loopRate = rospy.Rate(60)
        self.MainLoop()

    def SetAudioInfo(self, msg_info):
        self.channels = msg_info.channels
        self.sample_rate = msg_info.sample_rate
        self.sample_format = msg_info.sample_format
        self.bitrate = msg_info.bitrate
        self.coding_format = msg_info.coding_format
        if self.coding_format == "wave":
            fileSufix = ".wav"
        else:
            fileSufix = ".mp3"
        self.filePath = self.packagePath + "/resources/audios/tmp/raw_mic" + fileSufix

    def PackAudioAsMsg(self, file_path):
        msg_audio = AudioData()
        # print(type(self.audio_array[0].data))
        rospy.loginfo("[audio_tester] Packing file as AudioData msg")
        with open(file_path, 'rb') as file:
            msg_audio.data = file.read()
            rospy.loginfo("[audio_tester] Found {} bytes in AudioData msg".format(len(msg_audio.data)))
        return msg_audio
            
    def Callback_audio_info(self, msg):
        self.SetAudioInfo(msg)
   
    def Callback_audio(self, msg):
        t_now = rospy.get_time()
        dt = t_now - self.t_last

        if (dt > self.t_interval):
            print("-")

            rospy.loginfo("[audio_tester] Time since last file written: {0:.3f}s".format(dt))
            self.t_last = t_now

            rospy.loginfo("[audio_tester] Writing {} audio messages to {}".format(len(self.audio_array), self.filePath))
            self.WriteAudioData(self.filePath)

            self.pub_audio_preprocessed.publish(self.PackAudioAsMsg(self.filePath))

            rospy.loginfo("[audio_tester] Clearing buffer")
            self.audio_array = []

        else:
            self.audio_array.append(msg)

    # Writes audio to a file (decides file type)
    def WriteAudioData(self, file_path):
        if self.coding_format == "wave":
            self.WriteAudioDataToWav(file_path)
        elif self.coding_format == "mp3":
            self.WriteAudioDataToMp3(file_path)
        else:
            rospy.loginfo("[audio_tester] Invalid coding format")

    # Writes audio to .wav file
    def WriteAudioDataToWav(self, file_path):
        with wave.open(file_path, 'wb') as file:
            width = 2
            comptype = "NONE"
            compname = "noncompressed"
            file.setparams((self.channels, width, self.sample_rate, len(self.audio_array), comptype, compname))
            byte_count = 0
            for msg in self.audio_array:
                byte_count = byte_count + len(msg.data)
                file.writeframes(msg.data)
            rospy.loginfo("[audio_tester] Copied {} data bytes to file".format(byte_count))


    # Writes audio to .mp3 file
    def WriteAudioDataToMp3(self, file_path):
        with open(file_path, 'wb') as file:
            for msg in self.audio_array:
                file.write(msg.data)
    # Main loop
    def MainLoop(self):
        print("-")
        rospy.loginfo("[audio_tester] Looping")
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()

if __name__ == "__main__":
    TesterNode()
