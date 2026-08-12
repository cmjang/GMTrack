# Held-out cartwheel test data

`seed_cartwheel_heldout_unseen92_grounded.json` is the strict unseen test set.

- It contains 92 clips and 485.92 seconds.
- Its clip-name overlap with the 20-clip training set is exactly zero.
- Training 20 plus held-out 92 reconstructs the original 112-clip inventory.
- Four `cartwheelin` clips are retained here because none were used for training.

Use this manifest for generalization evaluation. The former combined 112-clip
manifest is not retained because it includes the 20 training clips and therefore is
not a valid unseen evaluation set.
