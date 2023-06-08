# Simple Stable Diffusion Web UI
# 简化版 Stable Diffusion Web UI

原版的[Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)实在是太复杂了，功能太多，安装中问题过多，很容易就爆了

这是一个**简化版的Stable Diffusion WebUI+API**，帮助你更好的在本地部署，使用UI、修改或调用接口。

- 删减了大量冗余代码，仅提供Linux的支持。

- API功能全部打开，无需启动时输入参数。

## 准备

你只需安装好Linux操作系统，conda环境，cuda，NV驱动等深度学习基础环境。

接下来创建并进入新的conda环境（py39+）

```bash
git clone https://github.com/uni-openai/stable-diffusion-simple.git
```

## 安装支持

```bash
pip3 install -r requirements.txt 
```

## 运行

```bash
python3 ./launch.py
```

## 关键接口

全部接口参考这个网址：<https://documenter.getpostman.com/view/9347507/2s93saZYEN>

1. [POST] `/sdapi/v1/txt2img`

```json
{
    "enable_hr": false,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "None",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "prompt": "a pretty girl",
    "styles": [],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "DPM++ SDE Karras",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": false,
    "tiling": false,
    "do_not_save_samples": false,
    "do_not_save_grid": false,
    "negative_prompt": "",
    "eta": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": true,
    "script_args": [],
    "sampler_index": "DPM++ SDE Karras",
    "script_name": "",
    "send_images": true,
    "save_images": false,
    "alwayson_scripts": {}
}
```

**说明**:

- 修改checkpoint model需要在options POST接口进行
