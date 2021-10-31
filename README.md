# chAInge
사진에서 배경에 존재하는 다른 사람들의 얼굴을 임의로 변환시켜주는 프로젝트

## Dependencies
- python 3.8 (tested on 3.8.0 / 3.8.10)
- pytorch 1.8.0
- torchvision 0.9.0
- torchaudio 0.8.0
- insightface 0.2.1
- onnxruntime 1.9.0
- pandas 1.2.0

## Usage
Download pre-trained models, [check here](./weights).

```python3 main.py [IMG_PATH]```, and result will be saved as ```images/output.jpg```

## References
- [FairFace](https://github.com/dchen236/FairFace)
- [SimSwap](https://github.com/neuralchen/SimSwap)