# Audio word to Semantic Vector
The project is used to convert an audio word to vector.

The project is divided to two part.

## 1st part
A audio word can be seen as a composition of phonetic vector and speaker vector.
#### This part is to seperate the two part from the original audio file.
The dataset we used is Librispeech for about 153GB.
The structure of this part can be found at [Towards Unsupervised Automatic Speech Recognition
Trained by Unaligned Speech and Text only](https://arxiv.org/pdf/1803.10952.pdf) Figure 1.
The difference between us and the paper is the loss function.
In our structure, we've jointly train the two encoder, decoder, discriminator and the speaker vector's L2 distance.

## 2nd part
#### Turn the phonetic part into the semantic vector.

