# KTNoiseReduction
基于WebRTC实现iOS端音频降噪功能

使用方法：
配置好路径就ok
    NSString * inpath = @“/ Users / apple / Desktop / a.wav;
    NSString *outpath = @"/Users/apple/Desktop/b.wav";
    
    
# 音频降噪算法 附完整C代码

降噪是音频图像算法中的必不可少的。
目的肯定是让图片或语音 更加自然平滑，简而言之，美化。
图像算法和音频算法 都有其共通点。
图像是偏向 空间 处理，例如图片中的某个区域。
图像很多时候是以二维数据为主，矩形数据分布。
音频更偏向 时间 处理，例如语音中的某段时长。
音频一般是一维数据为主，单声道波长。
处理方式也是差不多，要不单通道处理，然后合并，或者直接多通道处理。
只是处理时候数据参考系维度不一而已。
一般而言，
图像偏向于多通道处理，音频偏向于单通道处理。
而从数字信号的角度来看，也可以理解为聚类，频率归一化之类的。
总之就是对一些有一定规律的数字数据进行计算处理。
图像降噪被磨皮美颜这个大主题给带远了。
音频降噪目前感觉大有所为，像前面分享的《基于RNN的音频降噪算法 (附完整C代码)》
能达到这样的降噪效果，深度学习 确实有它独到的一面。
但是无可厚非，做机器学习之前还是要基于前人很多 基础算法进行数据的预处理等操作。
才能达到至善至美。
各有优劣，所谓算法肯定是互相配合为主，没有说谁能替换掉谁。
做算法最核心的思路就是使用各个算法的核心思想，放大它的优点，弱化它的缺点。
当然，做人也是如此。
音频降噪算法，网上公开的算法不多，资源也比较有限。
还是谷歌做了好事，把WebRTC开源，确实是一个基础。
前人种树，后人乘凉。
花了点时间，把WebRTC的噪声抑制模块提取出来，方便他人。
噪声抑制在WebRTC中有两个版本，一个是浮点，一个是定点。
一般定点做法是为了在一些特定环境下牺牲极少的精度，提升计算性能。
这个就不展开了，涉及到算法性能优化方面的一些知识点。
至于算法的实现，见源代码:
浮点版本:
noise_suppression.c 
定点版本:
noise_suppression_x.c
算法提供4个降噪级别，分别是：
enum nsLevel {
kLow,
kModerate,
kHigh,
kVeryHigh
};
实测效果还是很不错的，不过在一些特定的应用场景下，
其实这个算法还可以进一步调优。
改进思路，很多时候是基于需求来的,
打住打住，不细说了。
```
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//采用https://github.com/mackron/dr_libs/blob/master/dr_wav.h 解码
#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"
#include "noise_suppression.h"

#ifndef nullptr
#define nullptr 0
#endif

//写wav文件
void wavWrite_int16(char *filename, int16_t *buffer, size_t sampleRate, size_t totalSampleCount) {
drwav_data_format format = {};
format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
format.format = DR_WAVE_FORMAT_PCM;          // <-- Any of the DR_WAVE_FORMAT_* codes.
format.channels = 1;
format.sampleRate = (drwav_uint32) sampleRate;
format.bitsPerSample = 16;
drwav *pWav = drwav_open_file_write(filename, &format);
if (pWav) {
drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
drwav_uninit(pWav);
if (samplesWritten != totalSampleCount) {
fprintf(stderr, "ERROR\n");
exit(1);
}
}
}

//读取wav文件
int16_t *wavRead_int16(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
unsigned int channels;
int16_t *buffer = drwav_open_and_read_file_s16(filename, &channels, sampleRate, totalSampleCount);
if (buffer == nullptr) {
printf("读取wav文件失败.");
}
//仅仅处理单通道音频
if (channels != 1) {
drwav_free(buffer);
buffer = nullptr;
*sampleRate = 0;
*totalSampleCount = 0;
}
return buffer;
}

//分割路径函数
void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
const char *end;
const char *p;
const char *s;
if (path[0] && path[1] == ':') {
if (drv) {
*drv++ = *path++;
*drv++ = *path++;
*drv = '\0';
}
} else if (drv)
*drv = '\0';
for (end = path; *end && *end != ':';)
end++;
for (p = end; p > path && *--p != '\\' && *p != '/';)
if (*p == '.') {
end = p;
break;
}
if (ext)
for (s = end; (*ext = *s++);)
ext++;
for (p = end; p > path;)
if (*--p == '\\' || *p == '/') {
p++;
break;
}
if (name) {
for (s = p; s < end;)
*name++ = *s++;
*name = '\0';
}
if (dir) {
for (s = path; s < p;)
*dir++ = *s++;
*dir = '\0';
}
}

enum nsLevel {
kLow,
kModerate,
kHigh,
kVeryHigh
};

static float S16ToFloat_C(int16_t v) {
if (v > 0) {
return ((float) v) / (float) INT16_MAX;
}

return (((float) v) / ((float) -INT16_MIN));
}

void S16ToFloat(const int16_t *src, size_t size, float *dest) {
size_t i;
for (i = 0; i < size; ++i)
dest[i] = S16ToFloat_C(src[i]);
}

static int16_t FloatToS16_C(float v) {
static const float kMaxRound = (float) INT16_MAX - 0.5f;
static const float kMinRound = (float) INT16_MIN + 0.5f;
if (v > 0) {
v *= kMaxRound;
return v >= kMaxRound ? INT16_MAX : (int16_t) (v + 0.5f);
}

v *= -kMinRound;
return v <= kMinRound ? INT16_MIN : (int16_t) (v - 0.5f);
}

void FloatToS16(const float *src, size_t size, int16_t *dest) {
size_t i;
for (i = 0; i < size; ++i)
dest[i] = FloatToS16_C(src[i]);
}

int nsProcess(int16_t *buffer, size_t sampleRate, int samplesCount, enum nsLevel level) {
if (buffer == nullptr) return -1;
if (samplesCount == 0) return -1;
size_t samples = WEBRTC_SPL_MIN(160, sampleRate / 100);
if (samples == 0) return -1;
const int maxSamples = 320;
int num_bands = 1;
int16_t *input = buffer;
size_t nTotal = (samplesCount / samples);

NsHandle *nsHandle = WebRtcNs_Create();

int status = WebRtcNs_Init(nsHandle, sampleRate);
if (status != 0) {
printf("WebRtcNs_Init fail\n");
return -1;
}
status = WebRtcNs_set_policy(nsHandle, level);
if (status != 0) {
printf("WebRtcNs_set_policy fail\n");
return -1;
}
for (int i = 0; i < nTotal; i++) {
float inf_buffer[maxSamples];
float outf_buffer[maxSamples];
S16ToFloat(input, samples, inf_buffer);
float *nsIn[1] = {inf_buffer};   //ns input[band][data]
float *nsOut[1] = {outf_buffer};  //ns output[band][data]
WebRtcNs_Analyze(nsHandle, nsIn[0]);
WebRtcNs_Process(nsHandle, (const float *const *) nsIn, num_bands, nsOut);
FloatToS16(outf_buffer, samples, input);
input += samples;
}
WebRtcNs_Free(nsHandle);

return 1;
}

void noise_suppression(char *in_file, char *out_file) {
//音频采样率
uint32_t sampleRate = 0;
//总音频采样数
uint64_t inSampleCount = 0;
int16_t *inBuffer = wavRead_int16(in_file, &sampleRate, &inSampleCount);

//如果加载成功
if (inBuffer != nullptr) {
nsProcess(inBuffer, sampleRate, inSampleCount, kVeryHigh);
wavWrite_int16(out_file, inBuffer, sampleRate, inSampleCount);

free(inBuffer);
}
}

int main(int argc, char *argv[]) {
printf("WebRtc Noise Suppression\n");
printf("博客:http://cpuimage.cnblogs.com/\n");
printf("音频噪声抑制\n");
if (argc < 2)
return -1;
char *in_file = argv[1];
char drive[3];
char dir[256];
char fname[256];
char ext[256];
char out_file[1024];
splitpath(in_file, drive, dir, fname, ext);
sprintf(out_file, "%s%s%s_out%s", drive, dir, fname, ext);
noise_suppression(in_file, out_file);

printf("按任意键退出程序 \n");
getchar();
return 0;
}
