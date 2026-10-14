## Horizon vs tradability

| horizon | accuracy | prior_accuracy | directional_precision | trades | hit_rate | mid_pnl_per_trade | net_pnl_per_trade | net_pnl_total | t_stat |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.794 | 0.735 | 0.491 | 1509 | 0.011 | 0.299 | -2.445 | -3690.0 | -34.07 |
| 50 | 0.661 | 0.565 | 0.549 | 720 | 0.11 | 0.852 | -1.142 | -822.0 | -14.549 |
| 200 | 0.572 | 0.312 | 0.618 | 280 | 0.336 | 2.389 | 0.696 | 195.0 | 2.868 |

## Conformal coverage

| index | coverage | avg_set_size | singleton_rate | empty_rate | coverage_down | coverage_flat | coverage_up |
|---|---|---|---|---|---|---|---|
| Split conformal | 0.733 | 1.395 | 0.605 | 0.0 | 0.729 | 0.745 | 0.725 |
| Mondrian (class-conditional) | 0.733 | 1.399 | 0.601 | 0.0 | 0.782 | 0.729 | 0.679 |
| Adaptive (ACI, delayed labels) | 0.749 | 1.749 | 0.413 | 0.074 | 0.795 | 0.689 | 0.76 |

## Strategies

| index | signal_rate | trades | hit_rate | mid_pnl_per_trade | net_pnl_per_trade | net_pnl_total | t_stat |
|---|---|---|---|---|---|---|---|
| Always trade argmax | 0.664 | 271.0 | 0.369 | 2.629 | 1.077 | 292.0 | 4.455 |
| Confidence gate (tau=0.55, matched rate) | 0.441 | 239.0 | 0.381 | 2.812 | 1.126 | 269.0 | 3.912 |
| Conformal gate (alpha=0.25) | 0.448 | 232.0 | 0.431 | 3.17 | 1.621 | 376.0 | 6.021 |
| Adaptive conformal gate | 0.319 | 172.0 | 0.436 | 3.407 | 1.669 | 287.0 | 5.048 |
