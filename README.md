# Sorghum - Kaggle Contest
```

usage: train.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                [--momentum M] [--weight-decay W] [--print-freq N]
                [--resume PATH] [-e]
                DIR

Sordhum Training and Evaluation

positional arguments:
  DIR                   path to dataset

optional arguments:
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --clip-grad C         clip gradient (default: 0.1)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  
  ```
