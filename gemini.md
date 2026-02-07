This is our final year undergraduate research project.
We are trying to generate music using xLSTM and evaluate the long term structure of the generated music.
We are using lakhmidi dataset for training and evaluation.
As the baseline model, we are going to have the museformer model.

For training the xLSTM model, we are using the helibrunna repository.

Inside the /repo folder, you can find the related cloned repositories.
repos/helibrunna
repos/museformer
repos/MidiProcessor

Carefully read the README.md files of the repositories to understand the code.

In the museformer repository, they have not provided the pre-processed dataset. Instead, they have provided text files with the names of the midi files in the dataset that they have used for trainning, testing and validation.
I have extracted those midi files from the lakhmidi dataset, tokenized them using the REMIGEN2 format using MidiProcessor as mentioned in the museformer's README.md file.
The tokenized dataset is stored in the /data/lmd_preprocessed/splits folder. There are 3 files - train.txt, test.txt, valid.txt. Do not try to read the content of those files, they are very long. They are also gitignored. In those files, each line corresponds to one song.
Ex: part of a line - "s-9 o-42 t-38 i-21 p-67 d-1 v-31 o-45 t-38 i-21 p-67 d-1"

Then I used helibrunna to train the xLSTM model on this dataset. There we can specify the configuration for training in a yaml file inside the repos/helibrunna/configs folder.

I have trained a model with the configuration file repos/helibrunna/configs/lmd_remigen_xlstm_512d_2048ctx_12b.yaml.
The model is stored in the /repos/helibrunna/output/xlstm_lmd_512d_2048ctx_12b/run_20260126-0516. There are many checkpoints saved.

In the notebooks/xLSTM-2 folder, I have written a notebook to generate music using the trained model. Those are not working as much as I expected but they are a good starting point. 
There were few problems I faced when generating the music.
The model started generating some invalid token sequences and therefore the midi decoder failed to convert it to midi. Therefore, in the xlstm_music_generation.py file in the notebooks/xLSTM-2 folder, I have added a function to filter out the invalid token sequences. With that filtering the generated music sometimes get really messy.

But right now, I woul like to keep that aside for a while and focus on the evaluation of the model. I have the trained model and I still haven't use the train.txt and valid.txt for anything.


