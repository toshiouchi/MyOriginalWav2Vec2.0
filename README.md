# MyOriginalWav2Vec2.0
Wav2Vec2.0 modules with My Original Context Network

Changes of Context Network

The context network of the original Wav2Vec2.0 consists of Transoformer Encoder. On the other hand, a context network of my original Wav2Vec2.0 constists of Transformer Encoder, Downsampler and Transforemer Decoder. Inputs of Transformer Decoder are an output of Transformer Encoder as source input and a calculation amount by downsampling the Transformer Encoder output as target input.

A learning of pre-training

Data are 50 hours wav voice from JSUT 1.1 and Common Voice Japanese 11. A learning of 63 epochs about 20 hours with a RTX-A6000 were carried out.　A graph of loss change is 

![fig1](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/87ff5860-d6d4-4676-b361-b46117f6f84c)

Loss = Lm + α Ld ( α = 100 )

A learning of fine-tuning

Data are 7 horus wav voice from JSUT1.1 A learning of 167 epochs about 8 hours with a RTX-A6000 were carried out. A graph of loss change is

![Fig6](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/5eea5ab0-de50-4237-b22f-1763b57adb63)


A graph of train token error change is

![Fig7](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/a35baad5-b154-4b5e-aa4a-0f98a377691a)

Token error rate for train data is 5%.

A graph of vlidation token error change is

![Fig8](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/f3dd7ddc-d283-4ca0-a7a8-bc3516573c1f)

Token error rate of validation data is 41%.

## postscript

We carried out post calculation with Context Network with Transformer Encoder ( layers = 6, heads = 8, hidden_dim = 512, two convolution layer feed foward network with hiddem dim = (512, 2048) (2048,512 ) kernel_size = ( 5, 1  )) + Downsampler with downsampling_rate = 0.7 + Transformer Decoder with speculation similar to Transformer Encoder. 

### With Libri clean 360 hours data pre-training was carried out.

Calculations stopped due to a power outage that lasted about 0.5 seconds.

#### Chainging of Loss
![Fig30](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/be058392-4d24-4b09-854c-5ef1a114655f)

#### Changing of Similarity
![Fig31](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/2532bacf-8d74-4af2-9f55-30849f557bdb)

#### Changing of Lm
![Fig32](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/f1e49fdb-c6fb-425c-8fd0-0971aedfed07)

#### Changing of Ld 
![Fig33](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/448e5249-d1a6-4b6d-88d2-7e253ae47e2a)

#### Changing of Learning Rate
![Fig34](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/209a0ee7-db79-4459-8ad5-bbc14f98f3c7)

### With 10000 data in Libri clean 100 hours data fine tuning was carried out.

#### Changing of Loss
![Fig35](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/7446c65a-adad-40b5-a23f-4a7930907ef2)

Validation loss increased during the process, but validation token error rate did not increase, so learning continued.

#### Changing of Train Token Error Rate 
![Fig36](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/685e95fe-67ff-49b6-9d5c-9921527867e8)

#### Changing of Validation Toke  Error Rate
![Fig37](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/bfb9207c-74a3-4f09-b31f-3cacd94ecdbf)

T.E.R.　for validation data decreased to 17.9% level

#### Changing of Learning Rate
![Fig38](https://github.com/toshiouchi/MyOriginalWav2Vec2.0/assets/121741811/8570c523-2993-4e04-8604-0580bd47cbf9)
