# Summary of Vietnamese text with Transformers
For this project, I will use model Mixtral 8x7B and others technique like QLORA, DEEPSPEED and vLLM to increase training speed and decrease the hardware I use. I used 4xA100 SXM4 to train this model. And I use Gradio for create this app as an example of text summarization
# Environment configuration:
* Cuda: 11.8
* Cudnn: 8
* Python: 3.10.11
* Pytorch: 2.1.2
* Deepspeed: Stage 3 Offload
# Setup environment to run summarization app
* Run file `run_setup.sh -p` to setup environment for pip to run model and app to summarize content
* If you use anaconda or miniconda you should run file `run_setup.sh -c`
# Run summarization app
* Run file `run_app.sh` to run interface app for summarization
* When run app, this app will be create link public valid in 72 hours for everyone can access app