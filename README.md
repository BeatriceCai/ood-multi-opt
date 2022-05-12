# OOD-enhance-loss

### Environment
Pls see file `environment.txt`

### Implementation
 
- `train_OOD.py` is the main file. To run on MUV data: `python train_OOD.py --cwd='' --cwd_data='' --dataset='muv'`
- in `models.py`, `class multi_task_baseline` is the backbone model
- `ood_enhance_loss.py` is the core functionality to implement the proposed OOD_enhance_loss
- in `trainer.py`, `def trainer_alternative_1_loop()` present how to use OOD_enhance_loss in alternative training
