# Summary of Vietnamese text with Transformers
For this project, I will use model Mistral 7B and others technique like QLORA, DEEPSPEED and vLLM to increase training speed and decrease the hardware I use. I used 2xL40 with HOPPER architecture to train this model. And I use Gradio for create this app as an example of text summarization
# Environment configuration:
* Cuda: 11.8
* Cudnn: 8
* Python: 3.10.13
* Pytorch: 2.1.2
* Deepspeed: Stage 2 + Accelerate
# Setup environment to run summarization app
* Run file `run_setup.sh -p` to setup environment for pip to run model and app to summarize content
* If you use anaconda or miniconda you should run file `run_setup.sh -c`
* When you run error with library `mpi4py` you should remove folder `ld` in `compiler_compat` of `anaconda` or `miniconda`. For example my `anaconda` path in Linux (Ubuntu) is `/opt/conda`, you will run this command `rm /opt/conda/compiler_compat/ld` to fix error. And you run again `run_setup.sh -p`
# Run summarization app
* Run file `run_app.sh` to run interface app for summarization
* When run app, this app will be create link public valid in 72 hours for everyone can access app