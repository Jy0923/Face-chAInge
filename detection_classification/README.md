# Detection and Classification

Model detects people's faces in the picture and identify age, gender, and race.

- Face Detector : [RetinaFace](https://github.com/supernotman/RetinaFace_Pytorch/tree/8369b9304e19923c1a02c049df69628890bf30b5)
- Age,Gender,Race Classification : [FairFace](https://github.com/dchen236/FairFace)

<br>

## Pretrained Model

**You must Download the Pretrained Model** - model for [RetinaFace](https://www.dropbox.com/s/hvqveb6if724ise/model.pt?dl=0), [FairFace](https://drive.google.com/drive/folders/1B2gAnEpJ6oC9sMkcwS8v5Wk8PtHycHOV).

Download it and locate every *.pt* files in ```models``` directory.

For example, if you list up the ```models```, you should get result like this.

```bash
user@computer:~/chAInger/detection_classification$ ls models
fairface_alldata_4race_20191111.pt  model.pt  res34_fair_align_multi_7_20190809.pt
```

<br>

## Execute

You can put in any image.

```bash
python3 main_workflow.py [IMG_PATH]
```

The result image(commented with bounding box, face landmarks, cls labels) will be saved as "./out.jpg"
