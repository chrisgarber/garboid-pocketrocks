# Heuristic v1 benchmark

This report records the first fixed Monte Carlo evaluation of the Bayesian
aggressive, balanced, and passive heuristic bots.

## Environment and method

- Commit: `5454f41aaae23978907486622f40b6edd0b3e6b0`
  (`5454f41`, `test: benchmark heuristic bot tournament`)
- Python: 3.14.6
- Logical CPUs: 18
- Games per chart/player combination: 1,000
- Workers per command: 1
- Output format: JSON

The 15 chart/player jobs were launched concurrently. Each row's elapsed time is
therefore the real time of that individual process under concurrent load, not
an isolated timing. Games/second is `1,000 / real_seconds`; the individual
times must not be summed and interpreted as batch wall time.

The command template was:

```bash
/usr/bin/time -p env \
  UV_CACHE_DIR=/private/tmp/garboid-pocketrocks-uv-cache \
  uv run garboid-simulate \
    --bots BOT_LIST \
    --games 1000 \
    --players PLAYER_COUNT \
    --ruleset live-CHART \
    --seed ROOT_SEED \
    --workers 1 \
    --format json
```

The exact lineups and seeds were:

| Chart | Players | Seed | Lineup | Real (s) | Games/s |
|---|---:|---:|---|---:|---:|
| A | 3 | 45 | aggressive, balanced, passive | 30.66 | 32.616 |
| A | 4 | 46 | aggressive, balanced, passive, random | 30.08 | 33.245 |
| A | 5 | 47 | aggressive, balanced, passive, random, random | 31.34 | 31.908 |
| B | 3 | 145 | aggressive, balanced, passive | 30.62 | 32.658 |
| B | 4 | 146 | aggressive, balanced, passive, random | 30.07 | 33.256 |
| B | 5 | 147 | aggressive, balanced, passive, random, random | 31.24 | 32.010 |
| C | 3 | 245 | aggressive, balanced, passive | 30.73 | 32.541 |
| C | 4 | 246 | aggressive, balanced, passive, random | 30.25 | 33.058 |
| C | 5 | 247 | aggressive, balanced, passive, random, random | 31.52 | 31.726 |
| D | 3 | 345 | aggressive, balanced, passive | 30.38 | 32.916 |
| D | 4 | 346 | aggressive, balanced, passive, random | 29.95 | 33.389 |
| D | 5 | 347 | aggressive, balanced, passive, random, random | 31.15 | 32.103 |
| E | 3 | 445 | aggressive, balanced, passive | 30.45 | 32.841 |
| E | 4 | 446 | aggressive, balanced, passive, random | 30.14 | 33.179 |
| E | 5 | 447 | aggressive, balanced, passive, random, random | 31.36 | 31.888 |

The root seed is `42 + 100 * (ord(chart) - ord("A")) + players`.
Five-player games contain two instances of the same random bot specification.
Monte Carlo statistics aggregate by bot identity, so every five-player
`random` row represents 2,000 bot-games. The three heuristic rows each
represent 1,000 bot-games.

Rates below are percentages. Outright and tied-first rates use bot-games as
their denominator. A submitted bid of zero counts as a pass. Mean nonzero bid
excludes passes and zero bids. Action-win columns count resolved auction wins
and use these labels:

- `A1`: auction for one resource
- `A2`: auction for two resources
- `L10`: $10 loan
- `L20`: $20 loan
- `I5`: $5 investment
- `I10`: $10 investment

Counts are exact. Rates and means are rounded to three decimal places.

## Three-player results

| Chart | Bot | Bot-games | Outright | Tied first | Mean rank | Mean money | Pass rate | Mean nonzero bid | Objectives | Faults |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | aggressive | 1000 | 15.700% | 1.200% | 2.309 | 59.532 | 34.933% | 7.660 | 1581 | 0 |
| A | balanced | 1000 | 15.600% | 1.400% | 2.252 | 61.221 | 23.475% | 6.182 | 734 | 0 |
| A | passive | 1000 | 66.600% | 1.600% | 1.392 | 80.796 | 18.564% | 6.071 | 866 | 0 |
| B | aggressive | 1000 | 15.300% | 1.100% | 2.290 | 58.337 | 34.716% | 7.603 | 1740 | 0 |
| B | balanced | 1000 | 16.000% | 0.900% | 2.257 | 59.690 | 21.809% | 5.952 | 773 | 0 |
| B | passive | 1000 | 67.000% | 1.400% | 1.411 | 76.732 | 17.623% | 5.989 | 786 | 0 |
| C | aggressive | 1000 | 22.000% | 1.700% | 2.171 | 53.624 | 32.008% | 7.149 | 1740 | 0 |
| C | balanced | 1000 | 18.200% | 2.200% | 2.211 | 52.958 | 21.191% | 5.673 | 772 | 0 |
| C | passive | 1000 | 56.800% | 2.100% | 1.547 | 65.421 | 17.841% | 5.716 | 754 | 0 |
| D | aggressive | 1000 | 9.800% | 1.000% | 2.399 | 64.495 | 37.304% | 7.951 | 1508 | 0 |
| D | balanced | 1000 | 13.200% | 0.800% | 2.273 | 66.896 | 25.143% | 6.409 | 704 | 0 |
| D | passive | 1000 | 75.600% | 1.100% | 1.289 | 93.640 | 19.128% | 6.266 | 1007 | 0 |
| E | aggressive | 1000 | 16.500% | 0.900% | 2.245 | 63.699 | 36.008% | 7.664 | 1622 | 0 |
| E | balanced | 1000 | 14.200% | 1.000% | 2.315 | 62.423 | 23.852% | 6.229 | 654 | 0 |
| E | passive | 1000 | 68.000% | 0.700% | 1.393 | 86.383 | 18.410% | 6.127 | 983 | 0 |

| Chart | Bot | A1 | A2 | L10 | L20 | I5 | I10 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | aggressive | 2360 | 1698 | 931 | 912 | 288 | 332 |
| A | balanced | 2199 | 1066 | 364 | 97 | 494 | 248 |
| A | passive | 1899 | 1649 | 280 | 43 | 753 | 472 |
| B | aggressive | 2441 | 1656 | 889 | 910 | 309 | 371 |
| B | balanced | 2165 | 1109 | 366 | 73 | 516 | 207 |
| B | passive | 1800 | 1670 | 271 | 60 | 758 | 482 |
| C | aggressive | 2338 | 1858 | 773 | 894 | 332 | 372 |
| C | balanced | 2255 | 1115 | 385 | 80 | 530 | 253 |
| C | passive | 1825 | 1471 | 315 | 53 | 695 | 430 |
| D | aggressive | 2421 | 1527 | 979 | 906 | 278 | 352 |
| D | balanced | 1978 | 1078 | 350 | 78 | 558 | 260 |
| D | passive | 2056 | 1824 | 260 | 52 | 706 | 413 |
| E | aggressive | 2546 | 1631 | 908 | 891 | 271 | 319 |
| E | balanced | 1946 | 1076 | 349 | 80 | 542 | 275 |
| E | passive | 1917 | 1730 | 276 | 55 | 748 | 464 |

## Four-player results

| Chart | Bot | Bot-games | Outright | Tied first | Mean rank | Mean money | Pass rate | Mean nonzero bid | Objectives | Faults |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | aggressive | 1000 | 4.300% | 0.100% | 2.762 | 42.936 | 41.331% | 6.090 | 523 | 0 |
| A | balanced | 1000 | 12.900% | 0.400% | 2.241 | 52.197 | 22.152% | 5.544 | 520 | 0 |
| A | passive | 1000 | 80.000% | 0.500% | 1.222 | 76.742 | 17.773% | 5.416 | 1034 | 0 |
| A | random | 1000 | 2.300% | 0.000% | 3.728 | 13.661 | 26.570% | 6.820 | 470 | 0 |
| B | aggressive | 1000 | 4.900% | 0.600% | 2.621 | 41.160 | 39.559% | 5.736 | 728 | 0 |
| B | balanced | 1000 | 13.900% | 1.500% | 2.230 | 46.570 | 22.300% | 5.230 | 574 | 0 |
| B | passive | 1000 | 77.900% | 1.500% | 1.239 | 64.564 | 17.586% | 5.236 | 824 | 0 |
| B | random | 1000 | 1.400% | 0.200% | 3.844 | 7.378 | 27.873% | 6.831 | 480 | 0 |
| C | aggressive | 1000 | 4.700% | 0.600% | 2.640 | 39.851 | 37.000% | 5.716 | 645 | 0 |
| C | balanced | 1000 | 15.000% | 0.900% | 2.184 | 46.620 | 22.674% | 5.279 | 643 | 0 |
| C | passive | 1000 | 78.000% | 1.400% | 1.234 | 65.325 | 17.812% | 5.139 | 940 | 0 |
| C | random | 1000 | 0.800% | 0.100% | 3.875 | 4.423 | 27.763% | 6.686 | 361 | 0 |
| D | aggressive | 1000 | 2.900% | 0.400% | 2.796 | 44.179 | 41.633% | 6.252 | 547 | 0 |
| D | balanced | 1000 | 15.300% | 1.300% | 2.169 | 55.061 | 22.428% | 5.484 | 557 | 0 |
| D | passive | 1000 | 78.800% | 1.700% | 1.231 | 77.981 | 17.540% | 5.528 | 953 | 0 |
| D | random | 1000 | 1.300% | 0.000% | 3.751 | 15.385 | 27.283% | 6.808 | 424 | 0 |
| E | aggressive | 1000 | 3.000% | 0.200% | 2.808 | 41.848 | 44.105% | 6.373 | 522 | 0 |
| E | balanced | 1000 | 14.900% | 0.500% | 2.173 | 53.449 | 22.488% | 5.555 | 536 | 0 |
| E | passive | 1000 | 79.600% | 0.700% | 1.233 | 78.650 | 18.423% | 5.492 | 1116 | 0 |
| E | random | 1000 | 1.800% | 0.000% | 3.736 | 12.825 | 27.006% | 6.811 | 404 | 0 |

| Chart | Bot | A1 | A2 | L10 | L20 | I5 | I10 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | aggressive | 1168 | 899 | 57 | 74 | 144 | 219 |
| A | balanced | 1451 | 904 | 14 | 4 | 461 | 218 |
| A | passive | 1652 | 1572 | 11 | 0 | 585 | 409 |
| A | random | 1708 | 787 | 1358 | 894 | 221 | 128 |
| B | aggressive | 1227 | 978 | 57 | 61 | 114 | 256 |
| B | balanced | 1514 | 878 | 7 | 1 | 468 | 180 |
| B | passive | 1494 | 1481 | 7 | 0 | 613 | 423 |
| B | random | 1786 | 794 | 1391 | 942 | 224 | 122 |
| C | aggressive | 1091 | 941 | 60 | 77 | 189 | 267 |
| C | balanced | 1539 | 951 | 19 | 5 | 463 | 194 |
| C | passive | 1529 | 1535 | 25 | 1 | 612 | 415 |
| C | random | 1811 | 742 | 1343 | 874 | 177 | 113 |
| D | aggressive | 1183 | 891 | 75 | 87 | 157 | 229 |
| D | balanced | 1432 | 959 | 27 | 2 | 422 | 221 |
| D | passive | 1645 | 1603 | 7 | 1 | 603 | 346 |
| D | random | 1728 | 693 | 1335 | 839 | 258 | 149 |
| E | aggressive | 1246 | 841 | 49 | 80 | 142 | 185 |
| E | balanced | 1406 | 911 | 30 | 3 | 495 | 259 |
| E | passive | 1678 | 1622 | 23 | 2 | 601 | 390 |
| E | random | 1707 | 732 | 1375 | 881 | 227 | 145 |

## Five-player results

| Chart | Bot | Bot-games | Outright | Tied first | Mean rank | Mean money | Pass rate | Mean nonzero bid | Objectives | Faults |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | aggressive | 1000 | 2.300% | 0.200% | 3.181 | 34.453 | 50.807% | 5.654 | 332 | 0 |
| A | balanced | 1000 | 14.700% | 0.900% | 2.202 | 47.795 | 26.779% | 4.762 | 495 | 0 |
| A | passive | 1000 | 74.000% | 1.300% | 1.298 | 68.638 | 18.037% | 4.501 | 1007 | 0 |
| A | random | 2000 | 3.700% | 0.400% | 4.106 | 16.600 | 34.767% | 6.138 | 620 | 0 |
| B | aggressive | 1000 | 2.900% | 0.200% | 3.101 | 33.457 | 51.475% | 5.876 | 397 | 0 |
| B | balanced | 1000 | 15.000% | 1.000% | 2.188 | 45.928 | 25.599% | 4.723 | 545 | 0 |
| B | passive | 1000 | 77.200% | 1.100% | 1.260 | 64.166 | 17.611% | 4.550 | 919 | 0 |
| B | random | 2000 | 1.800% | 0.150% | 4.170 | 14.691 | 34.112% | 6.092 | 581 | 0 |
| C | aggressive | 1000 | 7.000% | 1.200% | 2.829 | 33.160 | 50.532% | 5.686 | 523 | 0 |
| C | balanced | 1000 | 17.000% | 2.000% | 2.180 | 40.511 | 25.648% | 4.626 | 519 | 0 |
| C | passive | 1000 | 69.400% | 2.500% | 1.343 | 55.511 | 18.269% | 4.328 | 856 | 0 |
| C | random | 2000 | 1.800% | 0.250% | 4.269 | 8.284 | 33.715% | 5.963 | 580 | 0 |
| D | aggressive | 1000 | 0.900% | 0.400% | 3.427 | 34.335 | 54.440% | 5.762 | 269 | 0 |
| D | balanced | 1000 | 14.600% | 1.000% | 2.187 | 53.473 | 28.107% | 5.011 | 447 | 0 |
| D | passive | 1000 | 77.800% | 1.500% | 1.257 | 79.226 | 18.678% | 4.589 | 1118 | 0 |
| D | random | 2000 | 2.550% | 0.200% | 4.019 | 21.469 | 34.503% | 6.260 | 613 | 0 |
| E | aggressive | 1000 | 3.100% | 0.100% | 3.179 | 35.590 | 48.977% | 5.539 | 371 | 0 |
| E | balanced | 1000 | 17.100% | 0.900% | 2.253 | 49.356 | 26.057% | 4.764 | 478 | 0 |
| E | passive | 1000 | 73.300% | 1.100% | 1.325 | 70.739 | 17.998% | 4.540 | 989 | 0 |
| E | random | 2000 | 2.550% | 0.350% | 4.079 | 17.993 | 34.373% | 6.130 | 593 | 0 |

| Chart | Bot | A1 | A2 | L10 | L20 | I5 | I10 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | aggressive | 727 | 738 | 6 | 5 | 139 | 252 |
| A | balanced | 1352 | 896 | 1 | 0 | 499 | 194 |
| A | passive | 1640 | 1459 | 0 | 0 | 525 | 377 |
| A | random | 2755 | 1314 | 1536 | 1012 | 429 | 250 |
| B | aggressive | 807 | 736 | 12 | 9 | 110 | 245 |
| B | balanced | 1313 | 951 | 1 | 0 | 469 | 182 |
| B | passive | 1556 | 1474 | 1 | 0 | 550 | 338 |
| B | random | 2733 | 1265 | 1500 | 982 | 428 | 237 |
| C | aggressive | 871 | 831 | 2 | 6 | 171 | 235 |
| C | balanced | 1248 | 922 | 0 | 0 | 493 | 213 |
| C | passive | 1358 | 1409 | 1 | 0 | 576 | 365 |
| C | random | 2900 | 1302 | 1532 | 1007 | 344 | 192 |
| D | aggressive | 641 | 740 | 5 | 9 | 78 | 210 |
| D | balanced | 1327 | 919 | 2 | 0 | 452 | 228 |
| D | passive | 1822 | 1512 | 1 | 1 | 535 | 337 |
| D | random | 2590 | 1273 | 1505 | 1020 | 491 | 305 |
| E | aggressive | 763 | 780 | 4 | 5 | 140 | 213 |
| E | balanced | 1354 | 887 | 1 | 1 | 481 | 213 |
| E | passive | 1651 | 1463 | 0 | 0 | 522 | 340 |
| E | random | 2663 | 1303 | 1561 | 988 | 437 | 260 |

## Fixed default regression

The default benchmark test runs the same 1,000-game live-A, three-player,
seed-42 tournament serially and with two workers, then requires exact result
equality. The serial-plus-two-worker test completed in 18.15 seconds, below the
30-second retention threshold.

| Bot | Bot-games | Outright | Tied first | Mean rank | Mean money | Pass rate | Mean nonzero bid | A1 | A2 | L10 | L20 | I5 | I10 | Objectives | Faults |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aggressive | 1000 | 12.200% | 0.600% | 2.407 | 57.286 | 34.662% | 7.551 | 2316 | 1616 | 825 | 882 | 268 | 299 | 1563 | 0 |
| balanced | 1000 | 17.400% | 0.900% | 2.192 | 61.716 | 22.146% | 6.110 | 2206 | 1121 | 378 | 123 | 578 | 240 | 703 | 0 |
| passive | 1000 | 69.000% | 1.300% | 1.358 | 80.618 | 18.055% | 6.103 | 1898 | 1701 | 345 | 51 | 692 | 473 | 895 | 0 |

The pass-rate spread is 16.606 percentage points and the mean-nonzero-bid
spread is $1.448. Aggressive won 1,707 loans (`L10 + L20`) versus passive's
396. All bot faults were zero, seat rotation was fair, and the serial and
two-worker results were identical.

## Findings and v1 profile decision

Passive dominated this self-play population in every chart/player
combination. Across its 15,000 bot-games it won outright 73.333% of the time,
with individual cells ranging from 56.800% to 80.000%. Its mean rank and mean
final money were also best in every cell. The effect was especially stark at
four and five players. These are matchup results against the specified lineup,
not evidence that passive is universally optimal.

Aggressive had the highest pass rate of the three heuristics in every cell.
Aggregated over 15,000 bot-games, the heuristic pass rates were:

| Profile | Bidding requests | Passes | Pass rate |
|---|---:|---:|---:|
| aggressive | 235079 | 99560 | 42.351720% |
| balanced | 235079 | 56455 | 24.015331% |
| passive | 235079 | 42532 | 18.092641% |

This seemingly counterintuitive result is consistent with the implemented
utility model: aggressive's high liquidity/time-value coefficient makes
spending or locking cash expensive, even though its low shading coefficient
makes accepted bids closer to its reservation value. As the field grows,
aggressive passes more and performs worse; in five-player cells it passed
48.977%-54.440% of bidding requests and won outright only 0.900%-7.000%.

Chart choice changed score levels and action mix but produced no faults or
chart-specific legality failure. Chart C generally depressed final money,
while chart D produced several of the highest heuristic money means. Tied
first rates remained low, topping out at 2.500%.

The required behavioral separation held in the fixed regression, so no
profile tuning was performed. The accepted v1 coefficients remain:

| Profile | Liquidity strength | Objective progress weight | Bid shading |
|---|---:|---:|---:|
| aggressive | 0.75 | 0.25 | 0.05 |
| balanced | 0.40 | 0.20 | 0.25 |
| passive | 0.15 | 0.15 | 0.50 |

The passive dominance and aggressive pass behavior are important baselines for
future opponent-aware bidding, coefficient tuning, and learned policies; this
report intentionally preserves them rather than retroactively tuning to make
the profiles equally strong.
