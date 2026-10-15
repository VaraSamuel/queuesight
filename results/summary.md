## Horizon vs tradability

| horizon | accuracy | prior_accuracy | directional_precision | trades | hit_rate | mid_pnl_per_trade | net_pnl_per_trade | net_pnl_total | t_stat |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.797 | 0.734 | 0.516 | 2974 | 0.011 | 0.32 | -2.67 | -7940.0 | -42.333 |
| 50 | 0.662 | 0.546 | 0.574 | 1493 | 0.139 | 0.978 | -0.999 | -1491.0 | -15.64 |
| 200 | 0.62 | 0.314 | 0.638 | 560 | 0.364 | 2.604 | 0.961 | 538.0 | 5.558 |

## Conformal coverage

| index | coverage | avg_set_size | singleton_rate | empty_rate | coverage_down | coverage_flat | coverage_up |
|---|---|---|---|---|---|---|---|
| Split conformal | 0.747 | 1.32 | 0.68 | 0.0 | 0.794 | 0.704 | 0.747 |
| Mondrian (class-conditional) | 0.749 | 1.327 | 0.673 | 0.0 | 0.739 | 0.745 | 0.764 |
| Adaptive (ACI, delayed labels) | 0.75 | 1.654 | 0.45 | 0.088 | 0.801 | 0.686 | 0.771 |

## Strategies

| index | signal_rate | trades | hit_rate | mid_pnl_per_trade | net_pnl_per_trade | net_pnl_total | t_stat |
|---|---|---|---|---|---|---|---|
| Always trade argmax | 0.673 | 560.0 | 0.364 | 2.604 | 0.961 | 538.0 | 5.558 |
| Confidence gate (tau=0.59, matched rate) | 0.454 | 521.0 | 0.382 | 2.826 | 1.161 | 605.0 | 6.017 |
| Conformal gate (alpha=0.25) | 0.459 | 518.0 | 0.382 | 2.823 | 1.12 | 580.0 | 6.184 |
| Adaptive conformal gate | 0.336 | 385.0 | 0.403 | 3.005 | 1.062 | 409.0 | 4.586 |
| Risk-controlled gate (LTT, loss rate <= 40%) | 0.513 | 544.0 | 0.36 | 2.589 | 0.882 | 480.0 | 4.692 |
