# AESRC2020-NTU

Data preparation scripts and training pipeline for the Accented English Speech Recognition.

Environment dependent
  1. Kaldi (Data preparation related function script) [Github link](https://github.com/kaldi-asr/kaldi)
  2. Espnet  [Githhub link](https://github.com/espnet/espnet)
  3. Google SentencePiece  [Github link](https://github.com/google/sentencepiece)
  
Instructions for use
  1. Data preparation
    All the data used in the experiment are stored in the `data` directory, in which train is used for training, valid is the verification set, cv_all and test are used for testing respectively.<br\>
    In order to better reproduce my experimental results, you can download the data set first, and then directly change the path in `wav.scp` in different sets in `data` directory.
    You can also use the `sed` command to replace the path in the wav.scp file with your path.<br\>
    Other files can remain unchanged, you can use it directly (eg, utt2IntLabel, utt2accent, text, utt2spk...).

  2. Single task system
    (1) Model file preparation
    `run_only_accent.sh` is used to train a single accent recognition model.<br\>
    Before running, you need to first put the model file(model/espnet/nets/pytorch_backend/e2e_asr_transformer_only_accent.py) to your espnet directory.<br\>
    eg:  `model/espnet/nets/pytorch_backend/e2e_asr_transformer_only_accent.py` to `/your espnet localtion/espnet/nets/pytorch_backend` <br\>

    (2) Step by step
    # feature extraction part
    bash run_only_accent.sh --nj 20 --steps 1-3
    # single accent recognition training
    bash run_only_accent.sh --nj 20 --steps 4
    # single accent recognition training and use asr Init, you should run the asr model first, and use asr model to replace `pretrained_model` variable in run_only_accent.sh
    bash run_only_accent.sh --nj 20 --steps 5
    
    
 
    
    
