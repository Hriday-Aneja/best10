SupeRR Selector S2 Round 1 - pressure model v3 (cricket-informed pressure definition)
Python 3: pandas, numpy, scipy, scikit-learn. Data CSVs in /home/claude/data (edit paths as needed).
 1. 01_clean_winprob.py        IPL + full-member T20I, rebuild runs/legal balls/state, win-probability model (Elo, venue par) -> balls_pressure_v4.pkl
 2. 02_pressure_definition.py  role-specific pressure: batting (chase demand vs venue par, behind par pace, collapse, flat-pitch tempo),
                               bowling (batting side on top, wickets in hand, death weighted, every phase) -> balls_pi.pkl
 3. 03_expected_values.py      context-expected runs / dismissals / wickets -> bat_v.pkl, bowl_v.pkl
 4. 08_grid.py                 held-out choice of shrinkage, pressure-shift penalty, pressure weight (jr.py, jr2.py)
 5. 09_final.py                final ratings, 20 sensitivity variants, 300-match bootstrap (fixed pool), out-of-sample validity, against-the-odds component (10%)
 6. 10_assemble.py             consensus ranking + evidence tables
09_final.py compares against the previous pressure definition via ../work2/*.pkl (remove that variant if not available).
