# ColonLight
This repository includes the testing code and a model checkpoint for lighting enhancement of colonoscopic frame sequences.

To do the lighting adjustment for a frame sequence, simply run 
```
bash test.sh
```
where `--chunk` is the path to original images. It will create a new folder `img_corr/` containing the enhanced images.
