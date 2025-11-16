uv run \
    -m src.model.test.mean_absolute_error \
    --testing-set "CHALEARN" \
    --model-path "checkpoints/IMDBWIKI_CHALEARN_epoch=11_val_loss=2.94.ckpt" \
    # --model-path "checkpoints/IMDBWIKI_APPAREAL_epoch=04_val_loss=2.93.ckpt" \

# This is our best recreation of DEX so far, with MAE of about 3.57 on ChaLearn LAP '15
# https://drive.google.com/drive/folders/1dFLn0RZ2FlvU8WjqwJ-YrOMor_SwTPfP?usp=sharing
# Note you will need to set transforms.Normalize() to std=MEAN_SUBTRACTION_ONLY
