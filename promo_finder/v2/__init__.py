"""v2 "Curiosity Engine": research-grounded promo finder.

annotate   - LLM labels every beat once (arousal, surprise, context need, turning point,
             questions opened/closed)           [Papalampidi et al. 2019/2021; Loewenstein 1994]
optimize   - code searches every clean clip for the best information gap
             at the cut                          [Loewenstein 1994; Wilmot & Keller 2020]
tournament - swapped-order pairwise LLM judging + Bradley-Terry ranking  [Zheng et al. 2023]
audience   - persona panel predicts the second each viewer swipes away   [Argyle et al. 2023]
"""
