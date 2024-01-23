# MyOriginalWav2Vec2.0
Wav2Vec2.0 modules with My Original Context Network

Changes of Context Network

The context network of the original Wav2Vec2.0 constitute of Transoformer Encoder, on the other hand a context network of my original Wav2Vec2.0 constitute of Transformer Encoder, Downsampler, Transforemer Decoder. Inputs of Transformer Decoder are an output of Transformer Encoder as source input and a calculation amount as target input by downsampling the Transformer Encoder output.

A learning of pre-training

Data are 50 hours wav voice from JSUT 1.1 and Common Voice Japanese 11. A learning of 63 epochs were carried out.　A graph of loss change is 
![fig1](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/87ff5860-d6d4-4676-b361-b46117f6f84c)

A learning of fine-tuning

Data are 7 horus wav voice from JSUT1.1 A learning of 167 epochs were carried out. A graph of train loss change is
![Fig7](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/a35baad5-b154-4b5e-aa4a-0f98a377691a)

WER is 5%.

A graph of vlidation loss cahge is
![Fig8](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/f3dd7ddc-d283-4ca0-a7a8-bc3516573c1f)

WER is 41%.
