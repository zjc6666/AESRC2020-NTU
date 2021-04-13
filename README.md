# AESRC2020-NTU

Data preparation scripts and training pipeline for the Accented English Speech Recognition.

Environment dependent
  1. Kaldi (Data preparation related function script) [Github link](https://github.com/kaldi-asr/kaldi)
  2. Espnet  [Githhub link](https://github.com/espnet/espnet)
  3. Google SentencePiece  [Github link](https://github.com/google/sentencepiece)
  
Instructions for use
  1. Data preparation
    All the data used in the experiment are stored in the `data` directory, in which train is used for training, valid is the verification set, cv_all and test are used for testing respectively.
    In order to better reproduce my experimental results, you can download the data set first, and then directly change the path in wav.scp in different data sets in data.
    You can also use the `sed` command to replace the path in the wav.scp file with your path.
