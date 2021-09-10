# Focused Attention Improves Document Grounded Generation
Code accompanying the paper: [Focused Attention Improves Document Grounded Generation](https://arxiv.org/pdf/2104.12714.pdf)

## Pre-requisites
- [transformers](https://github.com/huggingface/transformers) 3.1.0
- [NLG-Eval](https://github.com/Maluuba/nlg-eval)

## Data

### Wikipedia Update Generation Task
Download the raw data from this [link](https://github.com/shrimai/Towards-Content-Transfer-through-Grounded-Text-Generation).
The raw files have names ```train_info.csv, valid_info.csv, test_info.csv```. 
Use the ```scripts/prepare_wiki_update_gen.py``` script to prepare the data in the appropriate format with the command:

```
python scripts/prepare_wiki_update_gen.py --data_dir raw_data/ --out_dir data/wiki_update_gen/
```

### CMU DoG Task
Download the data from this [link](https://github.com/festvox/datasets-CMU_DoG).
Use the ```scripts/prepare_cmu_dog.py``` script to prepare the data in the appropriate format with the command:

```
python scripts/prepare_cmu_dog.py --data_dir datasets-CMU_DoG/ --out_dir data/cmu_dog/
```

## Quickstart

### BART Baseline
Use the ```run_train.py``` to train and test the BART baseline.
- Train the BART model using the following command:
```
python run_train.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_train --output_dir trained_models/wiki_update_gen/bart/ --log_file_path trained_models/wiki_update_gen/bart/log.txt --source_max_len 1024 --target_max_len 128
```

- Run the trained Bart model on the test set. This script creates two files `predictions.txt` and `reference.txt`, and saves it in the data_sir path provided.
```
python run_train.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_generate --output_dir trained_models/wiki_update_gen/bart/checkpoint-X/ --model_file_path trained_models/wiki_update_gen/bart/checkpoint-X/model.pt --source_max_len 1024 --target_max_len 128
```

### CoDR Model
Use the ```codr.py``` to train and test the CoDR baseline.
- Train the CoDR model using the following command:
```
python codr.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_train --output_dir trained_models/wiki_update_gen/codr/ --log_file_path trained_models/wiki_update_gen/codr/log.txt --source_max_len 1024 --target_max_len 128 --learning_rate 2e-5
```

- Run the trained CoDR model on the test set. This script creates two files `predictions.txt` and `reference.txt`, and saves it in the data_sir path provided.
```
python codr.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_generate --output_dir trained_models/wiki_update_gen/codr/checkpoint-X/ --model_file_path trained_models/wiki_update_gen/codr/checkpoint-X/model.pt --source_max_len 1024 --target_max_len 128
```

### DoHA Model
Use the ```doha.py``` to train and test the DoHA baseline.

**Important Note:** You have to copy the patch provided in ```patch``` folder to the desired location by running the ```apply_patch.sh``` script (You have to change the path where to copy this file). Find out the path where the transformers library is installed and replace the original ```generation_utils.py``` file in the transformers library with the ```patch/generation_utils.py``` file.
- Train the DoHA model using the following command:
```
python doha.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_train --output_dir trained_models/wiki_update_gen/doha/ --log_file_path trained_models/wiki_update_gen/doha/log.txt --source_max_len 1024 --target_max_len 128 --learning_rate 2e-5
```

- Run the trained DoHA model on the test set. This script creates two files `predictions.txt` and `reference.txt`, and saves it in the data_sir path provided.
```
python doha.py --data_dir data/wiki_update_gen/ --experiment_type 'chat_document' --do_generate --output_dir trained_models/wiki_update_gen/doha/checkpoint-X/ --model_file_path trained_models/wiki_update_gen/doha/checkpoint-X/model.pt --source_max_len 1024 --target_max_len 128
```

### Evaluation

```
nlg-eval --hypothesis=trained_models/wiki_update_gen/bart/checkpoint-X/predictions.txt --references=trained_models/wiki_update_gen/bart/checkpoint-X/reference.txt --no-skipthoughts --no-glove
```

## Trained Models
Download all the trained models from the links below. In each case, you will find three folders corresponding to the ```bart, codr and doha``` models, containing ```model.pt``` file.

```bash
http://tts.speech.cs.cmu.edu/document_grounded_generation/cmu_dog/cmu_dog.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/wikipedia_update_gen/wikipedia_update_gen.zip
```

Alternatively, if you are only interested in specific trained models, then you download the desired model from the links below:

```bash
http://tts.speech.cs.cmu.edu/document_grounded_generation/cmu_dog/cmu_dog_bart.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/cmu_dog/cmu_dog_codr.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/cmu_dog/cmu_dog_doha.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/wikipedia_update_gen/wikipedia_update_gen_bart.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/wikipedia_update_gen/wikipedia_update_gen_codr.zip
http://tts.speech.cs.cmu.edu/document_grounded_generation/wikipedia_update_gen/wikipedia_update_gen_doha.zip
```

## Contributors
If you use this code please cite the following:


    @inproceedings{prabhumoye-etal-2021-focused,
      title={Focused Attention Improves Document Grounded Generation},
      author={Prabhumoye, Shrimai and Hashimoto, Kazuma and Zhou, Yingbo and Black, Alan W and Salakhutdinov, Ruslan},
      booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics},
      publisher={Association for Computational Linguistics},
      year={2021},
      }
