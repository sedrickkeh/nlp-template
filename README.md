# NLP Template

This is a template for NLP projects in PyTorch. The sample task used is [Disaster Classification](https://www.kaggle.com/c/nlp-getting-started/overview). Data was split into train/valid/test and stored in `data`.


## Quickstart

### Training:
To train the model, run 
```bash
python train.py 
```
Config path is default set at `configs/config-train.yaml`, and extra parameters can be added (check `utils/config.py` for a list)

### Testing:
To test the model, run
```bash
python test.py --out_dir=(insert_out_dir_here) 
```
Here, the `out_dir` is the experiment directory storing the models created by the training function.


TODO:
- wandb integration