# Summary of Vietnamese text with Transformers
For this project, I will use model Mistral 7B and others technique like QLoRA, Deepspeed and Accelerate to increase training speed and decrease the hardware I use. To make it quick, I used 8xA100 SXM 80GB with AMPERE architecture and NVLINK to train this model. You can use only 1 GPU with 24 VRAM to train this model, and you must decrease batch size. And I use PEFT model and Gradio for create this app as an example of text summarization.
# Environment configuration
* Cuda: 12.1
* Cudnn: 8.9.7
* Python: 3.10.13
* Pytorch: 2.2.1+cu121
* Deepspeed: Stage 2 + Accelerate
# Dataset
I use dataset about Crawled Vietnamese News (Crawled with Selenium), Vietnamese News (VNDS), Vietnamese News Corpus (Binhvq News), ViMs, VLSP 2022. And I get 100,043 samples to train and 10,243 to test my model, which my training data includes 73,408 samples don't have title and 26,635 samples have title. I use title to train for adding information to context which I want to summary. And especially I only train with tile in ViMs dataset and I test it with no tile but ROUGE score still high.
|                          |Train     |Test      |Total     |
|:------------------------:|:--------:|:--------:|:--------:|
|Crawled Vietnamese News   |18,075    |2,009     |20,084    |
|VNDS                      |43,587    |4,769     |48,356    |
|Vietnamese News Corpus    |35,383    |2,949     |38,332    |
|ViMs                      |1,564     |297       |1,861     |
|VLSP AbMusu 2022          |1,434     |219       |1,653     |
|TOTAL                     |100,043   |10,243    |110,286   |
# Evaluation
I evaluate model with ROUGE score. But ROUGE score has the disadvantage that it cannot calculate scores between sentences with similar meanings.
|                          |ROUGE-1   |ROUGE-2   |ROUGE-L   |
|:------------------------:|:--------:|:--------:|:--------:|
|Crawled Vietnamese News   |18,075    |2,009     |20,084    |
|VNDS                      |43,587    |4,769     |48,356    |
|Vietnamese News Corpus    |35,383    |2,949     |38,332    |
|ViMs                      |1,564     |297       |1,861     |
|VLSP AbMusu 2022          |1,434     |219       |1,653     |
|AVG SCORE                 |100,043   |10,243    |110,286   |
# Setup environment to run summarization app
* To pull all this repository besides you use command `git pull` this, you should use command `git lfs pull` to get all file `LFS` in this repository. 
* Run file `run_setup.sh -p` to setup environment for pip to run model and app to summarize content.
* If you use anaconda or miniconda you should run file `run_setup.sh -c`.
* When you run error with library `mpi4py` you should remove folder `ld` in `compiler_compat` of `anaconda` or `miniconda`. For example my `anaconda` path in Linux (Ubuntu) is `/opt/conda`, you will run this command `rm /opt/conda/compiler_compat/ld` to fix error. And you run again `run_setup.sh -p`.
# Run summarization app
* Run file `run_app.sh` to run interface app for summarization.
* When run app, this app will be create link public valid in 72 hours (3 days) for everyone can access app.
