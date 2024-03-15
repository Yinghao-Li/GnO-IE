# GnO-IE

[![arXiv](https://img.shields.io/badge/arXiv-2402.13364-b31b1b.svg)](https://arxiv.org/abs/2402.13364)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Yinghao-Li/GnO-IE)

This repository provides the code, data, and results for the paper: [A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction](https://arxiv.org/abs/2402.13364).

The code is in `./src/`, data in `./data/`, and results in `./output/` and `./report/`, where `./output/` stores the conversation between the user and LLMs, and `./report/` stores the metrics and case studies.
The detailed structure is shown below.

```bash
.
├── data/  # the folder containing the data used in our project
│   ├── NER/  # Named Entity Recognition datasets
│   │   ├── ncbi/  # the NCBI dataset for NER
│   │   │   ├── train.json  # training data partition (not used)
│   │   │   ├── dev.json  # validation data partition (not used)
│   │   │   ├── test.json  # test data partition (used to evaluate our approach)
│   │   │   └── meta.json  # defines meta data such as entity types and LLM prompts
│   │   └── .../  # other datasets
│   └── RE/  # Relation Extraction datasets
│       └── .../  # has the same structure as NER datasets
├── output/  # the folder stores the LLM conversation history and generated training data for BERT
├── report/  # the folder stores the performance of each model/prompting method
├── resources/  # contains the user information for GPT Azure API
├── src/  # source code for G&O and baselines in our project
│   ├── core/  # package implementing the core functions, including arguments, communication with GPT, and metrics
│   ├── gpt/  # interaction with GPT for NER and RE tasks
│   ├── llama/  # inference of llama-based LLMs for NER and RE tasks
│   └── bert/  # BERT training and inference for NER
└── tasks/  # entry scripts of different tasks
    ├── ner_gpt.py  # interaction with GPT for NER
    ├── ner_llama.py  # interaction with llama-based LLMs for NER
    ├── re_gpt.py  # interaction with GPT for RE
    ├── re_llama.py  # interaction with llama-based LLMs for RE
    └── bert.py  # BERT training and inference for NER
```

## Requirements

This project is built with [Python 3.10](https://www.python.org).
For a complete list of required packages, please find them in the `requirements.txt` file.
To create a new `conda` environment and install the dependencies, you can use the following commands:

```bash
conda create -n gno python=3.10
conda activate gno

pip install -r requirements.txt
```

## Experiment Results

We have provided our experiment results and some ablation studies in the `./output/` and `./report/` directories.
The content in both directories is organized as `<NER|RE>/<model>/<prompting strategy>/dataset/`.

The `./output/` directory stores the conversation history between the user and LLM, with each file recording one data instance.
The numbers in file names represent the index of that particular data instance in the original test set in `./data/`.


The `./report/` directory contains the precision, recall, and f1 scores of each experiment.
In addition, you can also find true positive, false positive, and false negative cases for both partial match and full match recorded in the reports.

One exception is the supervised Transformer encoder fine-tuning (bert) experiment, which has reports stored in `./report/ber/ino/` but has no conversation history.
Instead, its training data is stored in `./output/data/bert/ino` as the training data for the DeBERTa V3 model.

## Run

If you are interested in reproducing our results or extending the experiment to a broader scope, you can use the Python scripts in the root directory.

Specifically, `./ner_gpt.py` targets NER task with GPT 3.5 model,`./ner_llama.py` targets NER task with Llama or Mistral models, etc.
`./bert.py` runs the fine-tuning and evaluation program of the supervised BERT-NER model.

To use GPT, you first need to fill the blanks in `./resources/gpt35.json` with your specific user information and credentials.
As an alternative, you can also modify the code in `./src/core/llm.py` to suit your needs.

Some example scripts are provided for you in `./scripts/`.
For example, You can run the GPT-NER code from the project root directory with
```bash
./scripts/ner_gpt.sh
```

You can also modify the shell scripts to your preference.
To check the meaning of each parameter, you can use
```bash
python ner_gpt.py --help
```

Notice that one difference between our code and the paper is the "G&O" method in the paper is represented by "ino" (instruct and organize) in the code.

To play with Llama or Mi[s/x]tral models, you also need to replace the model directory (`MODEL`) in `./scripts/ner_llama.sh` or `./scripts/re_llama.sh` files to the actual location where you store these models.
In addition, you may also need to change the `cuda_device` parameter.

## Citation

If you find our work useful, please consider citing it as
```
@article{Li-2024-GnO,
  title={A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction},
  author={Li, Yinghao and Ramprasad, Rampi and Zhang, Chao},
  journal={arXiv preprint arXiv:2402.13364},
  year={2024}
}
```
