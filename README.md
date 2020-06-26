
# LookaheadMinimax Optimizer

A PyTorch implementation of the extension of the Lookahead optimizer for GANs as introduced in [Taming GANs with Lookahead](https://arxiv.org/abs/2006.14567).

The original Lookahead optimizer's implementation can be found [here](https://github.com/michaelrzhang/lookahead).

## Usage

In PyTorch:
```python
G_optimizer = # {any optimizer} e.g. torch.optim.Adam
D_optimizer = # {any optimizer} e.g. torch.optim.Adam
if args.lookahead:
    G_optimizer = LookaheadMinimax(
        G_optimizer,
        D_optimizer,
        la_steps=args.la_steps,
        la_alpha=args.la_alpha,
        accumulate=args.num_accumulate 
        # ^ count multiple gradient accumulation steps as one lookahead step
    )

...

for _ in range(args.D_step_ratio):
    ...
    D_optimizer.step()

...

G_optimizer.step() # lookahead for BOTH optimizers happens here (every la_steps)
```

When loading optimizers from checkpoints, the state dict should be loaded using only the LookaheadMinimax optimizer. However, the LookaheadMinimax state_dict() function will return only the G_optimizer's state dict.
```python
if args.lookahead:
    G_optimizer.load_state_dict(checkpoint["G_optimizer"], checkpoint["D_optimizer"])
else:
    G_optimizer.load_state_dict(checkpoint["G_optimizer"])
    D_optimizer.load_state_dict(checkpoint["D_optimizer"])
```

Zhang et al. found that evaluation performance is typically better using the slow weights.
This can be done with something like this in your eval loop:
```python
if args.lookahead:
    optimizer._backup_and_load_cache()
    val_loss = eval_func(model)
    optimizer._clear_and_load_backup()
```