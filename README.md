# Summary of Vietnamese text with Transformers
For this project, I will use model Mistral 7B and others technique like QDoRA, Deepspeed and Accelerate to increase training speed and decrease the hardware I use. To make it quick, I used 8xA100 SXM 80GB with AMPERE architecture and NVLINK to train this model. You can use only 1 GPU with 24 VRAM to train this model, and you must decrease batch size. And I use PEFT model and Gradio for create this app as an example of text summarization.
# Environment configuration
* Cuda: 12.1
* Cudnn: 8.9.7
* Python: 3.10.14
* Pytorch: 2.3.0+cu121
* Deepspeed: Stage 2 + Accelerate
# Dataset
I use dataset about Vietnamese News Dataset (VNDS). This dataset have 99,134 samples to train and 22,348 to test my model. This dataset have deduplicated and clean. I got dataset to compare with model ViT5 base and ViT5 large in summarization task. And I want to compare the architecture encoder-decoder (ViT5) with decoder-only (Mistral 7B).
|                          |VNDS      |
|:------------------------:|:--------:|
|Train                     |99,134    |
|Test                      |22,498    |
|Total                     |121,632   |
# Evaluation
I evaluate model with ROUGE score. But ROUGE score has the disadvantage that it cannot calculate scores between sentences with similar meanings.
|                          |ROUGE-1   |ROUGE-2   |ROUGE-L   |
|:------------------------:|:--------:|:--------:|:--------:|
|ViT5 Base (256-length)    |61.85     |31.70     |41.70     |
|ViT5 Base (1024-length)   |62.77     |33.16     |42.75     |
|Vit5 Large (1024-length)  |63.37     |34.24     |43.55     |
|Mistral 7B + QDoRA        |70.12     |40.38     |52.18     |
# Setup environment to run summarization app
* To pull all this repository besides you use command `git pull` this, you should use command `git lfs pull` to get all file `LFS` in this repository. 
* Run file `run_setup.sh -p` to setup environment for pip to run model and app to summarize content.
* If you use anaconda or miniconda you should run file `run_setup.sh -c`.
* When you run error with library `mpi4py` you should remove folder `ld` in `compiler_compat` of `anaconda` or `miniconda`. For example my `anaconda` path in Linux (Ubuntu) is `/opt/conda`, you will run this command `rm /opt/conda/compiler_compat/ld` to fix error. And you run again `run_setup.sh -p`.
# Run summarization app
* Run file `run_app.sh` to run interface app for summarization.
* When run app, this app will be create link public valid in 72 hours (3 days) for everyone can access app.
