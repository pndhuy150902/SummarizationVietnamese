# Summary of Vietnamese text with Transformers
For this project, I will use model Mistral 7B and others technique like QLoRA, Deepspeed and Accelerate to increase training speed and decrease the hardware I use. To make it quick, I used 4xA100 80GB with AMPERE architecture and NVLINK to train this model. You can use only 1 GPU with 24 VRAM to train this model, and you must decrease batch size. And I use PEFT model and Gradio for create this app as an example of text summarization.
# Environment configuration
* Cuda: 12.1
* Cudnn: 8.9.7
* Python: 3.10.13
* Pytorch: 2.2.1+cu121
* Deepspeed: Stage 2 + Accelerate
# Dataset
I use dataset about Vietnamese News (VNDS), Crawled Vietnamese News (Crawled with Selenium), Vietnamese WikiHow, ViMs, VLSP 2022, Viet News Summarization (Huggingface). And I get 73,149 samples to train and 8,424 to test my model, which my training data includes 56,910 samples don't have title and 16,239 samples have title. I use title to train for adding information to context which I want to summary. And especially I only train with tile in ViMs dataset and I test it with no tile but ROUGE score still not low.
|                          |Train     |Test      |Total     |
|:------------------------:|:--------:|:--------:|:--------:|
|Vietnamese News Corpus    |22,614    |2,548     |25,162    |
|VNDS                      |21,297    |2,083     |23,380    |
|Wikilingua                |15,630    |2,098     |17,728    |
|Crawled Vietnamese News   |10,610    |1,179     |11,789    |
|ViMs                      |1,564     |297       |1,861     |
|VLSP 2022                 |1,434     |219       |1,653     |
|TOTAL                     |73,149    |8,424     |81,573    |
# Evaluation
|                          |ROUGE SCORE                     |
|                          |ROUGE-1   |ROUGE-2   |ROUGE-L   |
|:------------------------:|:--------:|:--------:|:--------:|
|Vietnamese News Corpus    |22,614    |2,548     |25,162    |
|VNDS                      |21,297    |2,083     |23,380    |
|Wikilingua                |15,630    |2,098     |17,728    |
|Crawled Vietnamese News   |10,610    |1,179     |11,789    |
|ViMs                      |1,564     |297       |1,861     |
|VLSP 2022                 |1,434     |219       |1,653     |
|AVG SCORE                 |73,149    |8,424     |81,573    |
# Setup environment to run summarization app
* To pull all this repository besides you use command `git pull` this, you should use command `git lfs pull` to get all file `LFS` in this repository. 
* Run file `run_setup.sh -p` to setup environment for pip to run model and app to summarize content.
* If you use anaconda or miniconda you should run file `run_setup.sh -c`.
* When you run error with library `mpi4py` you should remove folder `ld` in `compiler_compat` of `anaconda` or `miniconda`. For example my `anaconda` path in Linux (Ubuntu) is `/opt/conda`, you will run this command `rm /opt/conda/compiler_compat/ld` to fix error. And you run again `run_setup.sh -p`.
# Run summarization app
* Run file `run_app.sh` to run interface app for summarization.
* When run app, this app will be create link public valid in 72 hours for everyone can access app.
