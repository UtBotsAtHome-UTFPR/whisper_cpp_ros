// ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <audio_common_msgs/AudioData.h>

// Whisper
#include "whisper.h"

// Sound processing
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// C++
#include <thread>

typedef struct parameters {
    // Whisper parameters
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms  = 0;
    int32_t duration_ms  = 0;
    int32_t max_context  = -1;
    int32_t max_len      = 60;
    float word_thold = 0.01f;
    bool speed_up       = false;
    bool translate      = false;
    bool print_special  = false;
    bool print_progress = false;
    bool no_timestamps  = false;
    std::string language = "en";
    std::string model    = "ggml-base.bin";

    // Other parameters
    bool print_timings = false;
    bool show_result = true;
} parameters;

class WhisperNode
{
    // Whisper params
    parameters params;

    // Whisper context
    struct whisper_context* wsp_context;

    // ROS
    ros::NodeHandle nh;
    ros::Subscriber sub_audio_data = nh.subscribe("/voice/stt/audio_preprocessed", 10, &WhisperNode::CallbackAudioData, this);

    public:
        // Constructor
        WhisperNode()
        {
            InitParams();
            InitWhisper();
        }

        // Destructor
        ~WhisperNode()
        {
            whisper_free(wsp_context);
        }

        void InitParams()
        {
            ROS_INFO("[WHISPER] Initializing parameters");
            std::string package_path = ros::package::getPath("whisper_cpp_ros");
            nh.param("model", params.model, package_path + "/whisper.cpp/models/" + params.model);
        }

        void InitWhisper()
        {
            ROS_INFO("[WHISPER] system_info: n_threads = %d / %d | %s", params.n_threads * params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());
            ROS_INFO("[WHISPER] Initializing whisper");
            wsp_context = whisper_init(params.model.c_str());
            if (wsp_context == nullptr)
                ROS_INFO("[WHISPER] Error: failed to initialize whisper context");
        }

        // Callback function for audio data
        void CallbackAudioData(const audio_common_msgs::AudioData::ConstPtr &msg)
        {
            ROS_INFO("[WHISPER] - ");
            ROS_INFO("[WHISPER] Callback: AudioData (%d data bytes)", int(msg->data.size()));

            double t1 = ros::Time::now().toSec();
            std::vector<float> pcmf32;               // mono-channel F32 PCM
            std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
            ConvertMsgToWav(&(msg->data[0]), msg->data.size(), &pcmf32, &pcmf32s);
            Inference(&pcmf32);
            double t2 = ros::Time::now().toSec();
            double dt = t2 - t1;
            ROS_INFO("[WHISPER] Elapsed time: %f s", dt);
        }

        void ConvertMsgToWav(const void* data, size_t dataSize,
            std::vector<float>* pcmf32, std::vector<std::vector<float>>* pcmf32s)
        {
            ROS_INFO("[WHISPER] dr_wav loading wav from memory");
            drwav wav;

            // if (drwav_init_file(&wav, "/home/driver/apollo_ws/src/utbots_voice/utbots_voice/resources/audios/tmp/raw_mic.wav", NULL) == false)
            if (drwav_init_memory(&wav, data, dataSize, NULL) == false)
                ROS_INFO("[WHISPER] dr_wav failed to init wav");

            if (wav.channels != 1 && wav.channels != 2)
                ROS_INFO("[WHISPER] WAV must be mono or stereo\n");
            if (wav.sampleRate != WHISPER_SAMPLE_RATE)
                ROS_INFO("[WHISPER] WAV must be 16 kHz\n");
            if (wav.bitsPerSample != 16)
                ROS_INFO("[WHISPER] WAV must be 16-bit\n");

            const uint64_t n = wav.totalPCMFrameCount;

            std::vector<int16_t> pcm16;
            pcm16.resize(n * wav.channels);
            drwav_read_pcm_frames_s16(&wav, n, pcm16.data());

            // convert to mono, float
            pcmf32->resize(n);
            if (wav.channels == 1)
                for (uint64_t i = 0; i < n; i++)
                    (*pcmf32)[i] = float(pcm16[i])/32768.0f;
            else
                for (uint64_t i = 0; i < n; i++)
                    (*pcmf32)[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;

            ROS_INFO("[WHISPER] Processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s",
                int(pcmf32->size()), float(pcmf32->size())/WHISPER_SAMPLE_RATE, 
                params.n_threads, params.n_processors, params.language.c_str(), 
                params.translate ? "translate" : "transcribe");

            drwav_uninit(&wav);
        }

        void SetWhisperFullParams(whisper_full_params* wparams)
        {
            wparams->print_realtime   = false;
            wparams->print_progress   = params.print_progress;
            wparams->print_timestamps = !params.no_timestamps;
            wparams->print_special    = params.print_special;
            wparams->translate        = params.translate;
            wparams->language         = params.language.c_str();
            wparams->n_threads        = params.n_threads;
            wparams->n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams->n_max_text_ctx;
            wparams->offset_ms        = params.offset_t_ms;
            wparams->duration_ms      = params.duration_ms;

            wparams->token_timestamps = false;
            wparams->thold_pt         = params.word_thold;
            wparams->max_len          = params.max_len;

            wparams->speed_up         = params.speed_up;

            wparams->prompt_tokens    = nullptr;
            wparams->prompt_n_tokens  = 0;
        }

        void ShowResult()
        {
            const int n_segments = whisper_full_n_segments(wsp_context);
            std::string text("");
            for (int i = 0; i < n_segments; ++i)
                text = text + whisper_full_get_segment_text(wsp_context, i);
            ROS_INFO("[WHISPER] Result: %s", text.c_str());
        }

        void Inference(std::vector<float>* pcmf32)
        {
            ROS_INFO("[WHISPER] Setting full parameters");
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            SetWhisperFullParams(&wparams);

            ROS_INFO("[WHISPER] Inferencing");

            if (whisper_full_parallel(wsp_context, wparams, pcmf32->data(), pcmf32->size(), params.n_processors) != 0)
                ROS_INFO("[WHISPER] Failed to process audio");
            else 
            {
                ROS_INFO("[WHISPER] Inference done");
                if (params.show_result)
                    ShowResult();
                if (params.print_timings) 
                    whisper_print_timings(wsp_context);
            }
        }
};

int main(int argc, char **argv)
{
    // ROS
    ros::init(argc, argv, "whisper_node");
    WhisperNode* whisper_node = new WhisperNode();
    ros::Rate loopRate(30);

    ROS_INFO("[WHISPER] Looping");
    while (ros::ok())
    {
        loopRate.sleep();
        ros::spinOnce();
    }

    delete(whisper_node);
    return 0;
}