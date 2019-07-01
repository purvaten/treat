# Trick or TReAT : Thematic Reinforcement for Artistic Typography
This is an implementation of the ICCC '19 paper "Trick or TReAT : Thematic Reinforcement for Artistic Typography", by Purva Tendulkar, Kalpesh Krishna, Ramprasaath R. Selvaraju & Devi Parikh.

Full text available at: https://arxiv.org/abs/1903.07820

## Steps for Training
0. Create your primary directoy and enter it.

1. Create folders ```save```, ```args```, ```split```, to save the trained model, arguments, and train-val split respectively.

2. ```git clone https://github.com/purvaten/treat.git```

3. Train model
```
python treat/train.py \
--alpha 0.25 \
--data cliparts,letters \
--cliparts_dir treat/clipart_imgs \
--letters_dir treat/letter_imgs \
--datalimit 40250 \
--model alexnet
```
where ```treat/clipart_imgs``` and ```treat/letter_imgs``` should contain the clipart and letter images respectively for training. They each currently contain a single prototype image only.

**NOTE** : The letter images must be named in a specific format : `TYPE` + `LETTER` + `ID`, where `TYPE` = `upper` or `lower`,
`LETTER` = `a`, `b`, ... `z`, and `ID` is a number (optional). For example, `lowera1.png`, `upperb4.png`, etc.

## Results
| Word | Theme | TReAT |
| --- | --- | --- |
| storm | weather, disaster | ![storm](https://user-images.githubusercontent.com/13128829/60402621-156ad780-9b60-11e9-8d15-02111302c11e.png) |
| mouse | computer | ![mouse](https://user-images.githubusercontent.com/13128829/60402635-2c112e80-9b60-11e9-9d76-eb9b351ada71.png) |
| fish | ocean | ![fish](https://user-images.githubusercontent.com/13128829/60402638-3e8b6800-9b60-11e9-826b-dcf8459eebfc.png) |
| church | Jesus, God | ![church](https://user-images.githubusercontent.com/13128829/60402646-5bc03680-9b60-11e9-83d7-61012935cae8.png) |

**_NOTE_** : _The generated TReAT is independent of the sequence of provided themes._