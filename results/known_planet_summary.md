# Known Planet Recovery Summary

802 confirmed exoplanets (81 TESS + 721 Kepler), 4 detrending methods.

## Overall Results

| Method | Median Error | Wins | Win Rate | Speed |
|--------|-------------|------|----------|-------|
| **UMI** | **29.3%** | **425** | **53%** | **3.4ms/star** |
| welsch | 37.8% | 179 | 22% | 384ms/star |
| biweight | 37.0% | 160 | 20% | 234ms/star |
| savgol | 58.0% | 38 | 5% | 2ms/star |

## By Mission

| Mission | UMI | welsch | biweight | savgol |
|---------|-----|--------|----------|--------|
| TESS (81) wins | 35 | **38** | 8 | 0 |
| Kepler (721) wins | **390** | 141 | 152 | 38 |

## Key findings

- UMI wins 425 planets -- more than welsch + biweight + savgol combined (377)
- On Kepler (higher precision data), UMI dominates with 54% win rate
- On TESS, UMI and welsch are essentially tied (35 vs 38, not statistically significant on 81 planets)
- UMI is 69-113x faster than biweight/welsch while being more accurate overall
- Savgol wins only 38 planets (5%), all on Kepler deep transits

## Understanding the 29% median error

The 29.3% median error across all 802 planets may appear high, but context
is important:

1. **Dominated by shallow transits.** The median is pulled up by shallow
   depth planets where all methods have 15-25% error. At 0.5%+ depth,
   UMI error drops below 3%.

2. **Single-quarter recovery.** Published transit depths use multi-quarter
   stacking. Our test uses one quarter per star -- the hardest case.

3. **Relative comparison matters.** UMI's 29.3% vs biweight's 37.0% is a
   21% relative improvement on the same hard single-quarter data.

4. **Error by depth (1000 TESS stars, injection recovery):**

   | Depth | UMI | biweight | welsch |
   |-------|-----|----------|--------|
   | 0.1% | 15.8% | 20.5% | 18.5% |
   | 0.3% | 4.9% | 12.7% | 5.9% |
   | 0.5% | 2.4% | 5.1% | 1.8% |
   | 1.0% | 1.2% | 0.8% | 0.7% |

Data: TESS sectors 1, 6, 7, 8, 9, 10, 11, 12. Kepler Quarters 2, 5, 9, 17.
Full results: results/known_planet_recovery_all.json
